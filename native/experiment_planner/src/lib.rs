//! Experiment planner: which hardware test is worth running next (RSI1).
//!
//! One arm is one knob change (for example `readback_every=8`) applied to an already measured
//! configuration. Its effect is the log of the measured time ratio child / parent, so a
//! negative effect is a speed-up. Each arm keeps a conjugate normal posterior over its mean
//! effect (known measurement noise) and a running mean of what one test of it costs in
//! seconds. Observations update the state incrementally; nothing is ever refit from scratch.
//!
//! `choose` scores every eligible test by its expected improvement over the best measured
//! configuration, in percent points, per expected second, and stops when no test is worth
//! `min_gain_per_s`. The planner only orders hardware tests: whether a configuration is
//! correct and faster is decided by the caller's token identity and paired confirmation.
//!
//! Dependency free C ABI, no Python, model or GPU. The caller serializes access to a handle.

use std::slice;

pub const MAX_ARMS: usize = 256;
const MAGIC: &[u8; 10] = b"IEPSTATE01";

pub const OK: i32 = 0;
pub const ERR_NULL: i32 = -1;
pub const ERR_ARGUMENT: i32 = -2;
pub const ERR_CAPACITY: i32 = -3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannerConfig {
    /// Prior standard deviation of an arm's mean log effect.
    pub prior_sd: f64,
    /// Standard deviation of one measured log effect (screening noise).
    pub noise_sd: f64,
    /// Stop when the best test's expected improvement per second falls below this.
    pub min_gain_per_s: f64,
    /// Cost assumed for an arm that was never measured, in seconds.
    pub default_cost_s: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmStatus {
    pub observations: u64,
    pub failures: u64,
    pub posterior_mean: f64,
    pub posterior_sd: f64,
    pub cost_s: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Arm {
    n: u64,
    sum: f64,
    failures: u64,
    cost_n: u64,
    cost_mean: f64,
}

pub struct Planner {
    config: PlannerConfig,
    arms: [Arm; MAX_ARMS],
}

fn valid_config(c: &PlannerConfig) -> bool {
    c.prior_sd.is_finite() && c.prior_sd > 0.0 && c.noise_sd.is_finite() && c.noise_sd > 0.0
        && c.min_gain_per_s.is_finite() && c.min_gain_per_s >= 0.0
        && c.default_cost_s.is_finite() && c.default_cost_s > 0.0
}

impl Planner {
    pub fn new(config: PlannerConfig) -> Option<Planner> {
        valid_config(&config).then(|| Planner { config, arms: [Arm::default(); MAX_ARMS] })
    }

    /// Posterior mean and standard deviation of an arm's mean effect.
    pub fn posterior(&self, arm: usize) -> (f64, f64) {
        let a = &self.arms[arm];
        let prior_precision = 1.0 / (self.config.prior_sd * self.config.prior_sd);
        let noise_var = self.config.noise_sd * self.config.noise_sd;
        let precision = prior_precision + a.n as f64 / noise_var;
        ((a.sum / noise_var) / precision, (1.0 / precision).sqrt())
    }

    pub fn observe(&mut self, arm: usize, log_effect: f64, seconds: f64) -> i32 {
        if arm >= MAX_ARMS || !log_effect.is_finite() || !seconds.is_finite() || seconds < 0.0 {
            return ERR_ARGUMENT;
        }
        let a = &mut self.arms[arm];
        a.n += 1;
        a.sum += log_effect;
        Self::cost(a, seconds);
        OK
    }

    /// A test that could not run or whose tokens differed: it cost time and showed nothing usable.
    pub fn fail(&mut self, arm: usize, seconds: f64) -> i32 {
        if arm >= MAX_ARMS || !seconds.is_finite() || seconds < 0.0 {
            return ERR_ARGUMENT;
        }
        let a = &mut self.arms[arm];
        a.failures += 1;
        Self::cost(a, seconds);
        OK
    }

    fn cost(a: &mut Arm, seconds: f64) {
        a.cost_n += 1;
        a.cost_mean += (seconds - a.cost_mean) / a.cost_n as f64;
    }

    pub fn status(&self, arm: usize) -> ArmStatus {
        let a = &self.arms[arm];
        let (mean, sd) = self.posterior(arm);
        ArmStatus {
            observations: a.n,
            failures: a.failures,
            posterior_mean: mean,
            posterior_sd: sd,
            cost_s: if a.cost_n > 0 { a.cost_mean } else { self.config.default_cost_s },
        }
    }

    /// Expected improvement over `best_log`, in percent points, per expected second.
    pub fn score(&self, arm: usize, parent_log: f64, best_log: f64) -> f64 {
        let a = &self.arms[arm];
        if a.failures > a.n {
            return 0.0; // more failed than usable tests: not worth hardware time here
        }
        let (mean, sd) = self.posterior(arm);
        let m = parent_log + mean;
        let s = (sd * sd + self.config.noise_sd * self.config.noise_sd).sqrt();
        let z = (best_log - m) / s;
        let ei = (best_log - m) * normal_cdf(z) + s * normal_pdf(z);
        let cost = if a.cost_n > 0 { a.cost_mean.max(1e-3) } else { self.config.default_cost_s };
        100.0 * ei.max(0.0) / cost
    }

    /// Index into `arms` of the test to run next, or None when none is worth its time.
    pub fn choose(&self, arms: &[u32], parent_logs: &[f64], best_log: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (index, (&arm, &parent)) in arms.iter().zip(parent_logs).enumerate() {
            let value = self.score(arm as usize, parent, best_log);
            if best.map_or(true, |(_, v)| value > v) {
                best = Some((index, value));
            }
        }
        best.filter(|&(_, value)| value >= self.config.min_gain_per_s && value > 0.0)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(10 + 8 + MAX_ARMS * 40 + 8);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(MAX_ARMS as u64).to_le_bytes());
        for a in &self.arms {
            out.extend_from_slice(&a.n.to_le_bytes());
            out.extend_from_slice(&a.sum.to_le_bytes());
            out.extend_from_slice(&a.failures.to_le_bytes());
            out.extend_from_slice(&a.cost_n.to_le_bytes());
            out.extend_from_slice(&a.cost_mean.to_le_bytes());
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    pub fn from_bytes(config: PlannerConfig, input: &[u8]) -> Option<Planner> {
        let expected = 10 + 8 + MAX_ARMS * 40 + 8;
        if input.len() != expected || &input[..10] != MAGIC {
            return None;
        }
        let body = &input[..expected - 8];
        if fnv1a(body) != u64::from_le_bytes(input[expected - 8..].try_into().ok()?) {
            return None;
        }
        if u64::from_le_bytes(input[10..18].try_into().ok()?) != MAX_ARMS as u64 {
            return None;
        }
        let mut planner = Planner::new(config)?;
        let mut at = 18;
        let next = |at: &mut usize| -> [u8; 8] {
            let value: [u8; 8] = input[*at..*at + 8].try_into().unwrap();
            *at += 8;
            value
        };
        for a in planner.arms.iter_mut() {
            a.n = u64::from_le_bytes(next(&mut at));
            a.sum = f64::from_le_bytes(next(&mut at));
            a.failures = u64::from_le_bytes(next(&mut at));
            a.cost_n = u64::from_le_bytes(next(&mut at));
            a.cost_mean = f64::from_le_bytes(next(&mut at));
            if !a.sum.is_finite() || !a.cost_mean.is_finite() || a.cost_mean < 0.0 {
                return None;
            }
        }
        Some(planner)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Abramowitz and Stegun 7.1.26 through erf; absolute error below 1.5e-7.
fn normal_cdf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741
        + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + if x >= 0.0 { erf } else { -erf })
}

// -- C ABI ---------------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn iep_create(config: *const PlannerConfig) -> *mut Planner {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    Planner::new(*config).map_or(std::ptr::null_mut(), |p| Box::into_raw(Box::new(p)))
}

#[no_mangle]
pub unsafe extern "C" fn iep_free(handle: *mut Planner) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn iep_observe(handle: *mut Planner, arm: u32, log_effect: f64, seconds: f64) -> i32 {
    match handle.as_mut() {
        Some(p) => p.observe(arm as usize, log_effect, seconds),
        None => ERR_NULL,
    }
}

#[no_mangle]
pub unsafe extern "C" fn iep_fail(handle: *mut Planner, arm: u32, seconds: f64) -> i32 {
    match handle.as_mut() {
        Some(p) => p.fail(arm as usize, seconds),
        None => ERR_NULL,
    }
}

/// Writes the chosen index into `arms` (or -1 to stop) and its score.
#[no_mangle]
pub unsafe extern "C" fn iep_choose(handle: *const Planner, count: usize, arms: *const u32,
                                    parent_logs: *const f64, best_log: f64,
                                    out_index: *mut i64, out_score: *mut f64) -> i32 {
    let Some(p) = handle.as_ref() else { return ERR_NULL };
    if out_index.is_null() || out_score.is_null() || (count > 0 && (arms.is_null() || parent_logs.is_null())) {
        return ERR_NULL;
    }
    if !best_log.is_finite() {
        return ERR_ARGUMENT;
    }
    let (arms, parents) = if count == 0 {
        (&[][..], &[][..])
    } else {
        (slice::from_raw_parts(arms, count), slice::from_raw_parts(parent_logs, count))
    };
    if arms.iter().any(|&a| a as usize >= MAX_ARMS) || parents.iter().any(|v| !v.is_finite()) {
        return ERR_ARGUMENT;
    }
    match p.choose(arms, parents, best_log) {
        Some((index, score)) => {
            *out_index = index as i64;
            *out_score = score;
        }
        None => {
            *out_index = -1;
            *out_score = 0.0;
        }
    }
    OK
}

#[no_mangle]
pub unsafe extern "C" fn iep_status(handle: *const Planner, arm: u32, output: *mut ArmStatus) -> i32 {
    let Some(p) = handle.as_ref() else { return ERR_NULL };
    if output.is_null() {
        return ERR_NULL;
    }
    if arm as usize >= MAX_ARMS {
        return ERR_ARGUMENT;
    }
    *output = p.status(arm as usize);
    OK
}

#[no_mangle]
pub unsafe extern "C" fn iep_checkpoint(handle: *const Planner, output: *mut u8, capacity: usize,
                                        needed: *mut usize) -> i32 {
    let Some(p) = handle.as_ref() else { return ERR_NULL };
    if needed.is_null() {
        return ERR_NULL;
    }
    let bytes = p.to_bytes();
    *needed = bytes.len();
    if output.is_null() || capacity < bytes.len() {
        return ERR_CAPACITY;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len());
    OK
}

#[no_mangle]
pub unsafe extern "C" fn iep_restore(config: *const PlannerConfig, input: *const u8, length: usize) -> *mut Planner {
    if config.is_null() || input.is_null() {
        return std::ptr::null_mut();
    }
    Planner::from_bytes(*config, slice::from_raw_parts(input, length))
        .map_or(std::ptr::null_mut(), |p| Box::into_raw(Box::new(p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PlannerConfig {
        PlannerConfig { prior_sd: 0.05, noise_sd: 0.01, min_gain_per_s: 0.01, default_cost_s: 5.0 }
    }

    #[test]
    fn posterior_moves_towards_observations_incrementally() {
        let mut p = Planner::new(config()).unwrap();
        let (mean, sd) = p.posterior(3);
        assert!(mean == 0.0 && (sd - 0.05).abs() < 1e-12);
        for _ in 0..4 {
            assert_eq!(p.observe(3, -0.10, 2.0), OK);
        }
        let (mean, sd) = p.posterior(3);
        assert!(mean < -0.09 && mean > -0.10, "{mean}");
        assert!(sd < 0.01);
        assert_eq!(p.status(3).cost_s, 2.0);
        assert_eq!(p.observe(3, f64::NAN, 1.0), ERR_ARGUMENT);
        assert_eq!(p.observe(MAX_ARMS, 0.0, 1.0), ERR_ARGUMENT);
    }

    #[test]
    fn choose_prefers_a_learned_speedup_and_stops_when_nothing_pays() {
        let mut p = Planner::new(config()).unwrap();
        for _ in 0..5 {
            p.observe(1, -0.10, 2.0);
            p.observe(2, 0.02, 2.0);
        }
        assert_eq!(p.choose(&[2, 1], &[0.0, 0.0], 0.0), Some((1, p.score(1, 0.0, 0.0))));
        // Known to be slower and cheaper alternatives exhausted: stop.
        assert_eq!(p.choose(&[2], &[0.0], 0.0), None);
        // A test that keeps failing is not chosen.
        for _ in 0..3 {
            p.fail(4, 1.0);
        }
        assert_eq!(p.score(4, 0.0, 0.0), 0.0);
        assert_eq!(p.choose(&[], &[], 0.0), None);
    }

    #[test]
    fn unknown_arms_are_explored_by_their_uncertainty() {
        let p = Planner::new(config()).unwrap();
        assert!(p.choose(&[7], &[0.0], 0.0).is_some());
        let strict = Planner::new(PlannerConfig { min_gain_per_s: 100.0, ..config() }).unwrap();
        assert!(strict.choose(&[7], &[0.0], 0.0).is_none());
    }

    #[test]
    fn checkpoint_round_trips_and_rejects_corruption() {
        let mut p = Planner::new(config()).unwrap();
        p.observe(9, -0.03, 1.5);
        p.fail(10, 0.5);
        let bytes = p.to_bytes();
        let back = Planner::from_bytes(config(), &bytes).unwrap();
        assert_eq!(back.status(9), p.status(9));
        assert_eq!(back.status(10), p.status(10));
        let mut broken = bytes.clone();
        broken[40] ^= 1;
        assert!(Planner::from_bytes(config(), &broken).is_none());
        assert!(Planner::from_bytes(config(), &bytes[..bytes.len() - 1]).is_none());
        assert!(Planner::new(PlannerConfig { noise_sd: 0.0, ..config() }).is_none());
    }

    #[test]
    fn normal_cdf_is_accurate() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.959_964) - 0.975).abs() < 1e-6);
        assert!((normal_cdf(-1.959_964) - 0.025).abs() < 1e-6);
    }
}
