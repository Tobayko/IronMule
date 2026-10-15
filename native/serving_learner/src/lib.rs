//! RSI3-LIVE serving learner (backlog RSI3-LIVE): one numerical core for the live scheduler.
//!
//! Three parts, all updated from the same real transitions:
//!
//! * **Cost/transition model (AI).** Per action, a Bayesian linear regression predicts the group's
//!   service time and summed time to first token from the decision state, with a predictive
//!   variance. Each real observation is predicted before it is learned, so the reported error is
//!   the model's observed out-of-sample error, not its fit. A model whose observations fall outside
//!   two predictive standard deviations too often is retired (drift reset) and its support cleared.
//! * **Sequential RL.** Tabular Q-learning over a semi-Markov decision process: one decision per
//!   group dispatch, a time-based discount `exp(-dt / tau)`, and a target that bootstraps from the
//!   next decision state's value. A queue left behind by one action therefore costs that action
//!   through the next state's value, not only through its own requests.
//! * **Dreaming.** Dyna-style multi-step rollouts through the learned model from real start states,
//!   updating the same Q-table with a lower step size. Every rollout stops in a state/action pair
//!   without `min_support` real observations, when the model's relative uncertainty exceeds
//!   `max_rel_std`, when the simulated queue empties, or when the time budget is spent. Synthetic
//!   updates are counted separately and never count as support; only real transitions are stored.
//!
//! The reward is fixed by the caller's contract and computed here only: minus the request-seconds
//! spent in the system during the interval, plus `ttft_weight` x summed time to first token, plus
//! `failure_penalty_s` for every failed, cancelled, timed-out or rejected request. Dropping work is
//! never cheaper than serving it. Nothing here qualifies a policy: `isl_judge` applies the
//! optimizer's independent A/A/B rule (`ironmule_optimizer::qualify`) to measured window costs only.
//!
//! Std only; C ABI in `include/serving_learner.h`. The caller serializes calls on one handle.

use ironmule_optimizer::qualify::{judge_protocol, Rule, Triplet, Verdict};
use std::time::Instant;

pub const ABI: u32 = 3;
/// Waiting requests whose metadata a decision state carries (the widest catalogue action).
pub const WAITING: usize = 8;
pub const MAX_ACTIONS: usize = 12;
const Q_BINS: usize = 5;
const RATE_BINS: usize = 3;
const PROMPT_BINS: usize = 2;
pub const STATES: usize = Q_BINS * RATE_BINS * PROMPT_BINS;
const PHI: usize = 5;
const TARGETS: usize = 2; // 0: group service seconds, 1: seconds from dispatch to a member's first content
const MAX_RING: u32 = 4096;
const MAGIC: &[u8; 10] = b"ISLSTATE03";
const PRIOR: f64 = 1.0;
const DRIFT_WINDOW: f64 = 32.0;
const DRIFT_FLOOR: f64 = 0.7; // calibrated Gaussian errors sit near 0.95 inside two sigma

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IslConfig {
    pub abi_version: u32,
    pub action_count: u32,
    pub widths: [u32; MAX_ACTIONS],
    pub gather_s: [f64; MAX_ACTIONS],
    pub max_pending: u32,
    pub min_support: u32,
    pub ring_capacity: u32,
    pub horizon: u32,
    pub tau_s: f64,
    pub alpha_min: f64,
    pub epsilon: f64,
    pub synthetic_weight: f64,
    pub ttft_weight: f64,
    pub failure_penalty_s: f64,
    pub reward_scale_s: f64,
    pub max_rel_std: f64,
    pub seed: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IslState {
    pub queue: f64,
    pub arrival_rate: f64,
    pub prompt_chars: f64,
    pub max_tokens: f64,
    /// Over the oldest k = 1..=8 waiting requests (index k - 1): their summed age at the decision
    /// and summed prompt characters. Entries past the queue repeat the last value. Known now.
    pub age_cum: [f64; WAITING],
    pub chars_cum: [f64; WAITING],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IslTransition {
    pub state: IslState,
    pub next: IslState,
    pub action: u32,
    pub served: u32,
    pub ok: u32,
    pub bad: u32,
    /// 1 when every successful member has a first-content time (`first_exec_s` is then their
    /// mean); 0 when it is missing. A missing time is never read as zero (ABI 3).
    pub first_known: u32,
    pub duration_s: f64,
    pub service_s: f64,
    pub ttft_sum_s: f64,
    /// Mean over successful members of (first content - group dispatch): the executed part only.
    pub first_exec_s: f64,
    pub area_s: f64,
    pub unix_s: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IslDecision {
    pub action: u32,
    pub explored: u32,
    pub state_bin: u32,
    pub supported_mask: u32,
    pub q: [f64; MAX_ACTIONS],
    pub predicted_service_s: f64,
    pub predicted_service_std: f64,
    /// Seconds from dispatch to a member's first content (the model's part).
    pub predicted_first_s: f64,
    pub predicted_first_std: f64,
    /// Expected members, the known waiting part, and summed first content = known + members x first.
    pub expected_members: f64,
    pub known_wait_s: f64,
    pub predicted_ttft_sum_s: f64,
    pub predicted_ttft_std: f64,
    /// 1 when the state and action have `min_support` real observations; otherwise unusable.
    pub supported: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IslDreamResult {
    pub rollouts: u32,
    pub steps: u32,
    pub stopped_unsupported: u32,
    pub stopped_uncertain: u32,
    pub stopped_idle: u32,
    pub stopped_horizon: u32,
    pub budget_exhausted: u32,
    pub elapsed_ns: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IslStatus {
    pub ring_len: u64,
    pub real_updates: u64,
    pub synthetic_updates: u64,
    pub rollouts: u64,
    pub rollout_steps: u64,
    pub stopped_unsupported: u64,
    pub stopped_uncertain: u64,
    pub decisions: u64,
    pub explorations: u64,
    pub policy_changes: u64,
    pub drift_resets: u64,
    pub pruned: u64,
    pub model_observations: u64,
    pub mean_abs_rel_error: [f64; TARGETS],
    pub within_2sigma: f64,
    pub mean_reward: f64,
    pub last_td_error: f64,
    pub state_bytes: u64,
    /// Rollouts by the number of synthetic steps they took (0..=6, 7 = seven or more).
    pub depth_hist: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IslVerdict {
    /// 0 incomplete, 1 qualified, 2 rejected.
    pub status: u32,
    /// 0 none, 1 correctness, 2 regression, 3 noisy or unstable reference, 4 drift,
    /// 5 no qualified gain, 6 invalid measurement.
    pub reason: u32,
    pub mean_ratio: f64,
    pub mean_noise: f64,
    pub upper_bound: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Blr {
    a: [f64; PHI * PHI],
    b: [[f64; PHI]; TARGETS],
    sq: [f64; TARGETS],
    n: u64,
}

impl Blr {
    fn new() -> Blr {
        let mut a = [0.0; PHI * PHI];
        for i in 0..PHI {
            a[i * PHI + i] = PRIOR;
        }
        Blr { a, b: [[0.0; PHI]; TARGETS], sq: [0.0; TARGETS], n: 0 }
    }

    /// Mean and predictive variance per target. Before any data the variance is huge.
    fn predict(&self, phi: &[f64; PHI]) -> [(f64, f64); TARGETS] {
        let l = cholesky(&self.a);
        let v = solve(&l, phi); // A^-1 phi
        let lever: f64 = phi.iter().zip(&v).map(|(p, q)| p * q).sum();
        let mut out = [(0.0, 0.0); TARGETS];
        for (k, item) in out.iter_mut().enumerate() {
            let w = solve(&l, &self.b[k]);
            let mean: f64 = phi.iter().zip(&w).map(|(p, q)| p * q).sum();
            let noise = if self.n >= 2 { self.sq[k] / self.n as f64 } else { 1.0e4 };
            *item = (mean, noise.max(1e-9) * (1.0 + lever));
        }
        out
    }

    fn update(&mut self, phi: &[f64; PHI], y: &[f64; TARGETS], residual: &[f64; TARGETS]) {
        for i in 0..PHI {
            for j in 0..PHI {
                self.a[i * PHI + j] += phi[i] * phi[j];
            }
        }
        for k in 0..TARGETS {
            for i in 0..PHI {
                self.b[k][i] += phi[i] * y[k];
            }
            self.sq[k] += residual[k] * residual[k];
        }
        self.n += 1;
    }
}

fn cholesky(a: &[f64; PHI * PHI]) -> [f64; PHI * PHI] {
    let mut l = [0.0; PHI * PHI];
    for i in 0..PHI {
        for j in 0..=i {
            let mut s = a[i * PHI + j];
            for k in 0..j {
                s -= l[i * PHI + k] * l[j * PHI + k];
            }
            l[i * PHI + j] = if i == j { s.max(1e-12).sqrt() } else { s / l[j * PHI + j] };
        }
    }
    l
}

fn solve(l: &[f64; PHI * PHI], b: &[f64; PHI]) -> [f64; PHI] {
    let mut y = [0.0; PHI];
    for i in 0..PHI {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * PHI + k] * y[k];
        }
        y[i] = s / l[i * PHI + i];
    }
    let mut x = [0.0; PHI];
    for i in (0..PHI).rev() {
        let mut s = y[i];
        for k in i + 1..PHI {
            s -= l[k * PHI + i] * x[k];
        }
        x[i] = s / l[i * PHI + i];
    }
    x
}

#[derive(Clone, Debug, PartialEq)]
pub struct Learner {
    cfg: IslConfig,
    q: Vec<f64>,
    real_visits: Vec<u32>,
    synth_visits: Vec<u32>,
    support: Vec<u32>,
    models: Vec<Blr>,
    ring: Vec<IslTransition>,
    ring_head: usize,
    rng: u64,
    st: IslStatus,
    err_ewma: [f64; TARGETS],
    within_ewma: f64,
    drift_seen: f64,
}

fn valid_state(s: &IslState) -> bool {
    [s.queue, s.arrival_rate, s.prompt_chars, s.max_tokens].iter().chain(&s.age_cum).chain(&s.chars_cum)
        .all(|x| x.is_finite() && *x >= 0.0)
}

pub fn valid_config(c: &IslConfig) -> bool {
    let n = c.action_count as usize;
    c.abi_version == ABI
        && (1..=MAX_ACTIONS).contains(&n)
        && c.widths[..n].iter().all(|w| (1..=32).contains(w))
        && c.gather_s[..n].iter().all(|g| g.is_finite() && (0.0..=2.0).contains(g))
        && c.widths[n..].iter().all(|w| *w == 0)
        && c.gather_s[n..].iter().all(|g| *g == 0.0)
        && (1..=4096).contains(&c.max_pending)
        && (1..=1000).contains(&c.min_support)
        && (1..=MAX_RING).contains(&c.ring_capacity)
        && (1..=64).contains(&c.horizon)
        && c.tau_s.is_finite() && c.tau_s > 0.0
        && c.alpha_min.is_finite() && c.alpha_min > 0.0 && c.alpha_min <= 1.0
        && c.epsilon.is_finite() && (0.0..=1.0).contains(&c.epsilon)
        && c.synthetic_weight.is_finite() && (0.0..=1.0).contains(&c.synthetic_weight)
        && c.ttft_weight.is_finite() && c.ttft_weight >= 0.0
        && c.failure_penalty_s.is_finite() && c.failure_penalty_s > 0.0
        && c.reward_scale_s.is_finite() && c.reward_scale_s > 0.0
        && c.max_rel_std.is_finite() && c.max_rel_std > 0.0
}

impl Learner {
    pub fn new(cfg: IslConfig) -> Option<Learner> {
        if !valid_config(&cfg) {
            return None;
        }
        let cells = STATES * MAX_ACTIONS;
        Some(Learner {
            cfg,
            q: vec![0.0; cells],
            real_visits: vec![0; cells],
            synth_visits: vec![0; cells],
            support: vec![0; cells],
            models: (0..MAX_ACTIONS).map(|_| Blr::new()).collect(),
            ring: Vec::new(),
            ring_head: 0,
            rng: cfg.seed ^ 0x9e37_79b9_7f4a_7c15,
            st: IslStatus::default(),
            err_ewma: [0.0; TARGETS],
            within_ewma: 1.0,
            drift_seen: 0.0,
        })
    }

    fn actions(&self) -> usize {
        self.cfg.action_count as usize
    }

    pub fn bin(&self, s: &IslState) -> usize {
        let q = s.queue;
        let qb = if q <= 1.0 { 0 } else if q <= 2.0 { 1 } else if q <= 4.0 { 2 } else if q <= 8.0 { 3 } else { 4 };
        let rb = if s.arrival_rate < 0.1 { 0 } else if s.arrival_rate < 1.0 { 1 } else { 2 };
        let pb = usize::from(s.prompt_chars > 2000.0);
        (qb * RATE_BINS + rb) * PROMPT_BINS + pb
    }

    /// The group a decision expects, from what is known when it is made: the k = min(queue, width)
    /// oldest waiting requests, plus arrivals expected during the gather wait if the group is not
    /// full. Returns (k, expected members, expected prompt characters, known waiting seconds).
    fn expected(&self, s: &IslState, a: usize) -> (f64, f64, f64, f64) {
        let width = f64::from(self.cfg.widths[a]);
        let k = s.queue.max(1.0).min(width).floor();
        let index = (k as usize).clamp(1, WAITING) - 1;
        let gather = if s.queue < width { self.cfg.gather_s[a] } else { 0.0 };
        let joiners = (s.arrival_rate * gather).min(width - k).max(0.0);
        let mean_chars = s.chars_cum[index] / k;
        let known = s.age_cum[index] + k * gather + joiners * gather * 0.5;
        (k, k + joiners, s.chars_cum[index] + joiners * mean_chars, known)
    }

    /// Features per expected member count: a per-member prefill term (members, their prompt size).
    fn phi(&self, s: &IslState, a: usize) -> [f64; PHI] {
        let (_, n, chars, _) = self.expected(s, a);
        [1.0, n, n * n / 8.0, chars / 4096.0, s.max_tokens / 256.0]
    }

    /// The prediction for one state and action, with no side effect.
    pub fn predict(&self, s: &IslState, a: usize) -> Option<IslDecision> {
        if !valid_state(s) || a >= self.actions() {
            return None;
        }
        let b = self.bin(s);
        let p = self.models[a].predict(&self.phi(s, a));
        let (_, n, _, known) = self.expected(s, a);
        let mut d = IslDecision { action: a as u32, state_bin: b as u32, ..Default::default() };
        d.predicted_service_s = p[0].0;
        d.predicted_service_std = p[0].1.sqrt();
        d.predicted_first_s = p[1].0;
        d.predicted_first_std = p[1].1.sqrt();
        d.expected_members = n;
        d.known_wait_s = known;
        d.predicted_ttft_sum_s = known + n * p[1].0;
        d.predicted_ttft_std = n * p[1].1.sqrt();
        d.supported = u32::from(self.real_visits[b * MAX_ACTIONS + a] >= self.cfg.min_support
                                && self.support[b * MAX_ACTIONS + a] >= self.cfg.min_support);
        Some(d)
    }

    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn poisson(&mut self, lambda: f64) -> f64 {
        if !lambda.is_finite() || lambda <= 0.0 {
            return 0.0;
        }
        if lambda > 30.0 {
            return (lambda + lambda.sqrt() * self.normal()).round().max(0.0);
        }
        let limit = (-lambda).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }

    /// Minus request-seconds in the system, first-content seconds and penalties, scaled.
    pub fn reward(&self, area_s: f64, ttft_sum_s: f64, bad: f64) -> f64 {
        -(area_s + self.cfg.ttft_weight * ttft_sum_s + self.cfg.failure_penalty_s * bad) / self.cfg.reward_scale_s
    }

    fn value(&self, s: usize, mask: u32) -> Option<f64> {
        (0..self.actions())
            .filter(|a| mask & (1 << a) != 0 && self.real_visits[s * MAX_ACTIONS + a] > 0)
            .map(|a| self.q[s * MAX_ACTIONS + a])
            .fold(None, |best: Option<f64>, v| Some(best.map_or(v, |b| b.max(v))))
    }

    fn all_mask(&self) -> u32 {
        (1u32 << self.actions()) - 1
    }

    fn greedy(&self, s: usize, mask: u32, need_support: bool) -> usize {
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for a in 0..self.actions() {
            let i = s * MAX_ACTIONS + a;
            let known = if need_support { self.real_visits[i] >= self.cfg.min_support } else { self.real_visits[i] > 0 };
            if mask & (1 << a) == 0 || !known {
                continue;
            }
            if self.q[i] > best_v {
                best_v = self.q[i];
                best = a;
            }
        }
        best // the reference (0) when nothing is known
    }

    fn td(&mut self, s: usize, a: usize, r: f64, next: Option<usize>, dt: f64, weight: f64, real: bool) -> f64 {
        let gamma = (-dt.max(0.0) / self.cfg.tau_s).exp();
        let boot = match next {
            None => 0.0,
            Some(n) => self.value(n, self.all_mask()).unwrap_or_else(|| self.mean_q()),
        };
        let i = s * MAX_ACTIONS + a;
        let visits = f64::from(if real { self.real_visits[i] } else { self.real_visits[i] + self.synth_visits[i] });
        let alpha = (1.0 / (1.0 + visits).powf(0.6)).max(self.cfg.alpha_min) * weight;
        let delta = r + gamma * boot - self.q[i];
        self.q[i] += alpha * delta;
        delta
    }

    fn mean_q(&self) -> f64 {
        let (mut sum, mut n) = (0.0, 0.0);
        for (i, v) in self.real_visits.iter().enumerate() {
            if *v > 0 {
                sum += self.q[i];
                n += 1.0;
            }
        }
        if n > 0.0 { sum / n } else { 0.0 }
    }

    pub fn decide(&mut self, s: &IslState, mask: u32, explore: bool) -> Option<IslDecision> {
        if !valid_state(s) || mask & 1 == 0 || mask & !self.all_mask() != 0 {
            return None;
        }
        let b = self.bin(s);
        let mut d = IslDecision { state_bin: b as u32, ..Default::default() };
        for a in 0..self.actions() {
            d.q[a] = self.q[b * MAX_ACTIONS + a];
            if self.real_visits[b * MAX_ACTIONS + a] >= self.cfg.min_support {
                d.supported_mask |= 1 << a;
            }
        }
        let mut action = self.greedy(b, mask, false);
        if explore && self.uniform() < self.cfg.epsilon {
            // Directed: the eligible action with the fewest real observations in this state.
            let eligible: Vec<usize> = (0..self.actions()).filter(|a| mask & (1 << a) != 0).collect();
            let least = eligible.iter().map(|a| self.real_visits[b * MAX_ACTIONS + a]).min().unwrap_or(0);
            let ties: Vec<usize> = eligible.into_iter().filter(|a| self.real_visits[b * MAX_ACTIONS + a] == least).collect();
            action = ties[(self.next_u64() % ties.len() as u64) as usize];
            d.explored = 1;
            self.st.explorations += 1;
        }
        let p = self.predict(s, action)?;
        d.action = action as u32;
        d.predicted_service_s = p.predicted_service_s;
        d.predicted_service_std = p.predicted_service_std;
        d.predicted_first_s = p.predicted_first_s;
        d.predicted_first_std = p.predicted_first_std;
        d.expected_members = p.expected_members;
        d.known_wait_s = p.known_wait_s;
        d.predicted_ttft_sum_s = p.predicted_ttft_sum_s;
        d.predicted_ttft_std = p.predicted_ttft_std;
        d.supported = p.supported;
        self.st.decisions += 1;
        Some(d)
    }

    /// A completed real transition: model error, model update, Q update, ring storage.
    pub fn observe(&mut self, t: &IslTransition) -> bool {
        let a = t.action as usize;
        let finite = [t.duration_s, t.service_s, t.ttft_sum_s, t.first_exec_s, t.area_s, t.unix_s].iter().all(|x| x.is_finite() && *x >= 0.0);
        if a >= self.actions() || !finite || !valid_state(&t.state) || !valid_state(&t.next) || t.ok + t.bad > 100_000
            || t.first_known > 1 {
            return false;
        }
        let s = self.bin(&t.state);
        let phi = self.phi(&t.state, a);
        // A group with no successful member has no first content: it teaches the model nothing
        // (training it on zero would bias it), but its cost still reaches the Q-learner below.
        let informative = t.ok > 0 && t.first_known == 1;
        let before = self.greedy(s, self.all_mask(), true);
        if informative {
            let y = [t.service_s, t.first_exec_s];
            let pred = self.models[a].predict(&phi);
            let mut residual = [0.0; TARGETS];
            let mut inside = 0.0;
            for k in 0..TARGETS {
                residual[k] = y[k] - pred[k].0;
                let rel = residual[k].abs() / y[k].abs().max(1e-3);
                self.err_ewma[k] = if self.st.model_observations == 0 { rel } else { 0.9 * self.err_ewma[k] + 0.1 * rel };
                if residual[k].abs() <= 2.0 * pred[k].1.sqrt() {
                    inside += 0.5;
                }
            }
            self.within_ewma = if self.st.model_observations == 0 { inside } else { (1.0 - 1.0 / DRIFT_WINDOW) * self.within_ewma + inside / DRIFT_WINDOW };
            self.st.model_observations += 1;
            self.drift_seen += 1.0;
            self.models[a].update(&phi, &y, &residual);
            self.support[s * MAX_ACTIONS + a] += 1;
        }
        let r = self.reward(t.area_s, t.ttft_sum_s, f64::from(t.bad));
        let next = (t.next.queue >= 0.5).then(|| self.bin(&t.next));
        self.st.last_td_error = self.td(s, a, r, next, t.duration_s, 1.0, true);
        self.real_visits[s * MAX_ACTIONS + a] += 1;
        let n = self.st.real_updates as f64;
        self.st.mean_reward = (self.st.mean_reward * n + r) / (n + 1.0);
        self.st.real_updates += 1;
        if self.greedy(s, self.all_mask(), true) != before {
            self.st.policy_changes += 1;
        }
        self.push(*t);
        if informative && self.drift_seen >= DRIFT_WINDOW && self.within_ewma < DRIFT_FLOOR {
            // Retire the model: its predictions no longer cover what is observed.
            self.models = (0..MAX_ACTIONS).map(|_| Blr::new()).collect();
            self.support.iter_mut().for_each(|v| *v = 0);
            self.within_ewma = 1.0;
            self.drift_seen = 0.0;
            self.st.drift_resets += 1;
        }
        true
    }

    fn push(&mut self, t: IslTransition) {
        let cap = self.cfg.ring_capacity as usize;
        if self.ring.len() < cap {
            self.ring.push(t);
        } else {
            self.ring[self.ring_head] = t;
            self.ring_head = (self.ring_head + 1) % cap;
        }
    }

    /// Drop stored transitions older than `before_unix_s` (retention); learned values stay.
    pub fn prune(&mut self, before_unix_s: f64) -> u32 {
        let mut ordered: Vec<IslTransition> = self.ring[self.ring_head..].iter().chain(&self.ring[..self.ring_head]).copied().collect();
        let len = ordered.len();
        ordered.retain(|t| t.unix_s >= before_unix_s);
        let removed = (len - ordered.len()) as u32;
        self.ring = ordered;
        self.ring_head = 0;
        if self.ring.len() == self.cfg.ring_capacity as usize {
            self.ring_head = 0;
        }
        self.st.pruned += u64::from(removed);
        removed
    }

    pub fn dream(&mut self, max_rollouts: u32, budget_ns: u64) -> IslDreamResult {
        let start = Instant::now();
        let mut out = IslDreamResult::default();
        let over = |start: &Instant| start.elapsed().as_nanos() as u64 >= budget_ns;
        while out.rollouts < max_rollouts && !self.ring.is_empty() {
            if over(&start) {
                out.budget_exhausted = 1;
                break;
            }
            let pick = (self.next_u64() % self.ring.len() as u64) as usize;
            let mut s = self.ring[pick].state;
            out.rollouts += 1;
            self.st.rollouts += 1;
            let mut h = 0;
            loop {
                if h >= self.cfg.horizon {
                    out.stopped_horizon += 1;
                    break;
                }
                if over(&start) {
                    out.budget_exhausted = 1;
                    break;
                }
                let b = self.bin(&s);
                let supported: Vec<usize> = (0..self.actions())
                    .filter(|a| self.real_visits[b * MAX_ACTIONS + a] >= self.cfg.min_support && self.support[b * MAX_ACTIONS + a] >= self.cfg.min_support)
                    .collect();
                if supported.is_empty() {
                    out.stopped_unsupported += 1;
                    self.st.stopped_unsupported += 1;
                    break;
                }
                let mask = supported.iter().fold(0u32, |m, a| m | (1 << a));
                let a = if self.uniform() < self.cfg.epsilon {
                    supported[(self.next_u64() % supported.len() as u64) as usize]
                } else {
                    self.greedy(b, mask, true)
                };
                let p = self.models[a].predict(&self.phi(&s, a));
                let (service_mean, service_var) = p[0];
                // Summed first content of the simulated group: its members x the model's first-content
                // time (waiting is already in the area term). Same uncertainty rule as service time.
                let members = s.queue.max(1.0).min(f64::from(self.cfg.widths[a]));
                let (ttft_mean, ttft_var) = (members * p[1].0, members * members * p[1].1);
                if service_mean <= 0.0 || service_var.sqrt() > self.cfg.max_rel_std * service_mean
                    || ttft_var.sqrt() > self.cfg.max_rel_std * ttft_mean.abs().max(service_mean) {
                    out.stopped_uncertain += 1;
                    self.st.stopped_uncertain += 1;
                    break;
                }
                let service = (service_mean + service_var.sqrt() * self.normal()).max(0.05 * service_mean);
                let ttft = (ttft_mean + ttft_var.sqrt() * self.normal()).max(0.0);
                let q = s.queue.max(1.0);
                let served = q.min(f64::from(self.cfg.widths[a]));
                let dt = self.cfg.gather_s[a] + service;
                let arrivals = self.poisson(s.arrival_rate * dt);
                let mut next_q = q - served + arrivals;
                let rejected = (next_q - f64::from(self.cfg.max_pending)).max(0.0);
                next_q -= rejected;
                let area = q * dt + 0.5 * arrivals * dt;
                let r = self.reward(area, ttft, rejected);
                let pick = (self.next_u64() % self.ring.len() as u64) as usize;
                let template = self.ring[pick].next;
                // The simulated next state: queue and rate from the model; waiting metadata from a real
                // next state (ages are not simulated, so they stay those of a real queue of that size).
                let next = IslState { queue: next_q, arrival_rate: s.arrival_rate, ..template };
                let nb = (next_q >= 0.5).then(|| self.bin(&next));
                let weight = self.cfg.synthetic_weight;
                self.td(b, a, r, nb, dt, weight, false);
                self.synth_visits[b * MAX_ACTIONS + a] += 1;
                self.st.synthetic_updates += 1;
                self.st.rollout_steps += 1;
                out.steps += 1;
                h += 1;
                if nb.is_none() {
                    out.stopped_idle += 1;
                    break;
                }
                s = next;
            }
            self.st.depth_hist[(h as usize).min(7)] += 1;
        }
        out.elapsed_ns = start.elapsed().as_nanos() as u64;
        out
    }

    /// The frozen candidate: per state, the best action with `min_support` real observations, else
    /// the reference. Synthetic updates shape the values; they never make an action eligible.
    pub fn policy(&self) -> [u32; STATES] {
        let mut out = [0u32; STATES];
        for (s, slot) in out.iter_mut().enumerate() {
            *slot = self.greedy(s, self.all_mask(), true) as u32;
        }
        out
    }

    pub fn status(&self) -> IslStatus {
        let mut st = self.st;
        st.ring_len = self.ring.len() as u64;
        st.mean_abs_rel_error = self.err_ewma;
        st.within_2sigma = self.within_ewma;
        st.state_bytes = self.encoded_len() as u64;
        st
    }

    /// `encode().len()` without encoding (status is polled; the state is not small).
    pub fn encoded_len(&self) -> usize {
        let cells = STATES * MAX_ACTIONS;
        let config = 4 + 4 + MAX_ACTIONS * 4 + MAX_ACTIONS * 8 + 4 * 4 + 8 * 8 + 8;
        let model = (PHI * PHI + TARGETS * PHI + TARGETS) * 8 + 8;
        let transition = 8 * (4 + 2 * WAITING) * 2 + 4 * 5 + 8 * 6;
        MAGIC.len() + config + cells * 8 + 3 * cells * 4 + MAX_ACTIONS * model + 16 + self.ring.len() * transition + 8 + 12 * 8 + 6 * 8 + 8 * 8
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer(Vec::with_capacity(64 * 1024));
        w.0.extend_from_slice(MAGIC);
        write_config(&mut w, &self.cfg);
        for v in &self.q { w.f(*v); }
        for v in self.real_visits.iter().chain(&self.synth_visits).chain(&self.support) { w.u(*v); }
        for m in &self.models {
            for v in m.a { w.f(v); }
            for row in m.b { for v in row { w.f(v); } }
            for v in m.sq { w.f(v); }
            w.l(m.n);
        }
        w.l(self.ring.len() as u64);
        w.l(self.ring_head as u64);
        for t in &self.ring { write_transition(&mut w, t); }
        w.l(self.rng);
        let s = &self.st;
        for v in [s.real_updates, s.synthetic_updates, s.rollouts, s.rollout_steps, s.stopped_unsupported, s.stopped_uncertain,
                  s.decisions, s.explorations, s.policy_changes, s.drift_resets, s.pruned, s.model_observations] {
            w.l(v);
        }
        for v in [s.mean_reward, s.last_td_error, self.err_ewma[0], self.err_ewma[1], self.within_ewma, self.drift_seen] {
            w.f(v);
        }
        for v in s.depth_hist { w.l(v); }
        w.0
    }

    pub fn decode(cfg: IslConfig, bytes: &[u8]) -> Option<Learner> {
        let mut l = Learner::new(cfg)?;
        let mut r = Reader { b: bytes, i: 0 };
        if r.take(MAGIC.len())? != MAGIC || read_config(&mut r)? != cfg {
            return None;
        }
        for v in l.q.iter_mut() { *v = r.f()?; }
        for v in l.real_visits.iter_mut().chain(l.synth_visits.iter_mut()).chain(l.support.iter_mut()) { *v = r.u()?; }
        for m in l.models.iter_mut() {
            for v in m.a.iter_mut() { *v = r.f()?; }
            for row in m.b.iter_mut() { for v in row.iter_mut() { *v = r.f()?; } }
            for v in m.sq.iter_mut() { *v = r.f()?; }
            m.n = r.l()?;
        }
        let len = r.l()? as usize;
        let head = r.l()? as usize;
        if len > cfg.ring_capacity as usize || (head != 0 && head >= len) {
            return None;
        }
        for _ in 0..len {
            l.ring.push(read_transition(&mut r)?);
        }
        l.ring_head = head;
        l.rng = r.l()?;
        let s = &mut l.st;
        for v in [&mut s.real_updates, &mut s.synthetic_updates, &mut s.rollouts, &mut s.rollout_steps, &mut s.stopped_unsupported,
                  &mut s.stopped_uncertain, &mut s.decisions, &mut s.explorations, &mut s.policy_changes, &mut s.drift_resets,
                  &mut s.pruned, &mut s.model_observations] {
            *v = r.l()?;
        }
        s.mean_reward = r.f()?;
        s.last_td_error = r.f()?;
        l.err_ewma = [r.f()?, r.f()?];
        l.within_ewma = r.f()?;
        l.drift_seen = r.f()?;
        for v in l.st.depth_hist.iter_mut() { *v = r.l()?; }
        let finite_models = l.models.iter().all(|m| m.a.iter().chain(m.b.iter().flatten()).chain(&m.sq).all(|v| v.is_finite()));
        if r.i != bytes.len() || !l.q.iter().all(|v| v.is_finite()) || !finite_models {
            return None;
        }
        Some(l)
    }
}

struct Writer(Vec<u8>);
impl Writer {
    fn f(&mut self, v: f64) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn u(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn l(&mut self, v: u64) { self.0.extend_from_slice(&v.to_le_bytes()); }
}

struct Reader<'a> { b: &'a [u8], i: usize }
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.i.checked_add(n)?;
        let out = self.b.get(self.i..end)?;
        self.i = end;
        Some(out)
    }
    fn f(&mut self) -> Option<f64> { Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?)) }
    fn u(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    fn l(&mut self) -> Option<u64> { Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?)) }
}

fn write_config(w: &mut Writer, c: &IslConfig) {
    w.u(c.abi_version);
    w.u(c.action_count);
    for v in c.widths { w.u(v); }
    for v in c.gather_s { w.f(v); }
    for v in [c.max_pending, c.min_support, c.ring_capacity, c.horizon] { w.u(v); }
    for v in [c.tau_s, c.alpha_min, c.epsilon, c.synthetic_weight, c.ttft_weight, c.failure_penalty_s, c.reward_scale_s, c.max_rel_std] {
        w.f(v);
    }
    w.l(c.seed);
}

fn read_config(r: &mut Reader) -> Option<IslConfig> {
    let abi_version = r.u()?;
    let action_count = r.u()?;
    let mut widths = [0u32; MAX_ACTIONS];
    for v in widths.iter_mut() { *v = r.u()?; }
    let mut gather_s = [0.0; MAX_ACTIONS];
    for v in gather_s.iter_mut() { *v = r.f()?; }
    Some(IslConfig {
        abi_version, action_count, widths, gather_s,
        max_pending: r.u()?, min_support: r.u()?, ring_capacity: r.u()?, horizon: r.u()?,
        tau_s: r.f()?, alpha_min: r.f()?, epsilon: r.f()?, synthetic_weight: r.f()?, ttft_weight: r.f()?,
        failure_penalty_s: r.f()?, reward_scale_s: r.f()?, max_rel_std: r.f()?, seed: r.l()?,
    })
}

fn write_state(w: &mut Writer, s: &IslState) {
    for v in [s.queue, s.arrival_rate, s.prompt_chars, s.max_tokens] { w.f(v); }
    for v in s.age_cum.iter().chain(&s.chars_cum) { w.f(*v); }
}

fn read_state(r: &mut Reader) -> Option<IslState> {
    let mut s = IslState { queue: r.f()?, arrival_rate: r.f()?, prompt_chars: r.f()?, max_tokens: r.f()?, ..Default::default() };
    for v in s.age_cum.iter_mut() { *v = r.f()?; }
    for v in s.chars_cum.iter_mut() { *v = r.f()?; }
    Some(s)
}

fn write_transition(w: &mut Writer, t: &IslTransition) {
    write_state(w, &t.state);
    write_state(w, &t.next);
    for v in [t.action, t.served, t.ok, t.bad, t.first_known] { w.u(v); }
    for v in [t.duration_s, t.service_s, t.ttft_sum_s, t.first_exec_s, t.area_s, t.unix_s] { w.f(v); }
}

fn read_transition(r: &mut Reader) -> Option<IslTransition> {
    Some(IslTransition {
        state: read_state(r)?, next: read_state(r)?, action: r.u()?, served: r.u()?, ok: r.u()?, bad: r.u()?,
        first_known: r.u()?,
        duration_s: r.f()?, service_s: r.f()?, ttft_sum_s: r.f()?, first_exec_s: r.f()?, area_s: r.f()?, unix_s: r.f()?,
    })
}

/// The optimizer's qualification verdict as plain numbers.
pub fn verdict(protocol: u32, rule: &Rule, triplets: &[Triplet]) -> IslVerdict {
    match judge_protocol(protocol, rule, triplets) {
        Verdict::Incomplete => IslVerdict { status: 0, ..Default::default() },
        Verdict::Qualified { mean_ratio, mean_noise, upper_bound } => IslVerdict { status: 1, reason: 0, mean_ratio, mean_noise, upper_bound },
        Verdict::Rejected { reason, mean_ratio, mean_noise, upper_bound } => {
            let code = if reason == "correctness failed" { 1 } else if reason.starts_with("regression") { 2 }
                else if reason.starts_with("noisy") || reason.starts_with("unstable") { 3 }
                else if reason.starts_with("invalid: drift") { 4 } else if reason == "no qualified gain" { 5 } else { 6 };
            IslVerdict { status: 2, reason: code, mean_ratio, mean_noise, upper_bound }
        }
    }
}

// ---- C ABI ----

/// # Safety
/// `config` must point to a valid `IslConfig`.
#[no_mangle]
pub unsafe extern "C" fn isl_create(config: *const IslConfig) -> *mut Learner {
    match config.as_ref().and_then(|c| Learner::new(*c)) {
        Some(l) => Box::into_raw(Box::new(l)),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `handle` must come from `isl_create`/`isl_restore` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn isl_free(handle: *mut Learner) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// Valid pointers; calls on one handle never overlap.
#[no_mangle]
pub unsafe extern "C" fn isl_decide(handle: *mut Learner, state: *const IslState, eligible_mask: u32, explore: u32, out: *mut IslDecision) -> i32 {
    let (Some(l), Some(s), Some(o)) = (handle.as_mut(), state.as_ref(), out.as_mut()) else { return 1 };
    if explore > 1 {
        return 1;
    }
    match l.decide(s, eligible_mask, explore == 1) {
        Some(d) => { *o = d; 0 }
        None => 1,
    }
}

/// Prediction for one action without any side effect (no ticket, RNG or learning).
/// # Safety
/// Valid pointers.
#[no_mangle]
pub unsafe extern "C" fn isl_predict(handle: *const Learner, state: *const IslState, action: u32, out: *mut IslDecision) -> i32 {
    let (Some(l), Some(s), Some(o)) = (handle.as_ref(), state.as_ref(), out.as_mut()) else { return 1 };
    match l.predict(s, action as usize) {
        Some(d) => { *o = d; 0 }
        None => 1,
    }
}

/// # Safety
/// Valid pointers; calls on one handle never overlap.
#[no_mangle]
pub unsafe extern "C" fn isl_observe(handle: *mut Learner, transition: *const IslTransition) -> i32 {
    let (Some(l), Some(t)) = (handle.as_mut(), transition.as_ref()) else { return 1 };
    if l.observe(t) { 0 } else { 1 }
}

/// # Safety
/// Valid pointers; calls on one handle never overlap.
#[no_mangle]
pub unsafe extern "C" fn isl_dream(handle: *mut Learner, max_rollouts: u32, budget_ns: u64, out: *mut IslDreamResult) -> i32 {
    let (Some(l), Some(o)) = (handle.as_mut(), out.as_mut()) else { return 1 };
    if l.ring.is_empty() {
        *o = IslDreamResult::default();
        return 3;
    }
    *o = l.dream(max_rollouts, budget_ns);
    0
}

/// # Safety
/// `out` must hold `STATES` entries.
#[no_mangle]
pub unsafe extern "C" fn isl_policy(handle: *const Learner, out: *mut u32) -> i32 {
    let Some(l) = handle.as_ref() else { return 1 };
    if out.is_null() {
        return 1;
    }
    std::ptr::copy_nonoverlapping(l.policy().as_ptr(), out, STATES);
    0
}

/// # Safety
/// Valid pointers. Returns `STATES` for an invalid state.
#[no_mangle]
pub unsafe extern "C" fn isl_bin(handle: *const Learner, state: *const IslState) -> u32 {
    match (handle.as_ref(), state.as_ref()) {
        (Some(l), Some(s)) if valid_state(s) => l.bin(s) as u32,
        _ => STATES as u32,
    }
}

/// # Safety
/// Valid pointers.
#[no_mangle]
pub unsafe extern "C" fn isl_status(handle: *const Learner, out: *mut IslStatus) -> i32 {
    let (Some(l), Some(o)) = (handle.as_ref(), out.as_mut()) else { return 1 };
    *o = l.status();
    0
}

/// # Safety
/// Valid pointers.
#[no_mangle]
pub unsafe extern "C" fn isl_prune(handle: *mut Learner, before_unix_s: f64) -> u32 {
    match handle.as_mut() {
        Some(l) if before_unix_s.is_finite() => l.prune(before_unix_s),
        _ => 0,
    }
}

/// Returns 5 with `needed` set when `capacity` is too small.
/// # Safety
/// `output` holds `capacity` bytes; `needed` is valid.
#[no_mangle]
pub unsafe extern "C" fn isl_checkpoint(handle: *const Learner, output: *mut u8, capacity: usize, needed: *mut usize) -> i32 {
    let (Some(l), Some(n)) = (handle.as_ref(), needed.as_mut()) else { return 1 };
    let bytes = l.encode();
    *n = bytes.len();
    if output.is_null() || capacity < bytes.len() {
        return 5;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len());
    0
}

/// Null for a corrupt checkpoint or one written under another configuration.
/// # Safety
/// `config` valid; `input` holds `length` bytes.
#[no_mangle]
pub unsafe extern "C" fn isl_restore(config: *const IslConfig, input: *const u8, length: usize) -> *mut Learner {
    let Some(c) = config.as_ref() else { return std::ptr::null_mut() };
    if input.is_null() {
        return std::ptr::null_mut();
    }
    match Learner::decode(*c, std::slice::from_raw_parts(input, length)) {
        Some(l) => Box::into_raw(Box::new(l)),
        None => std::ptr::null_mut(),
    }
}

/// The independent qualification rule on `count` measured A/B/AA window costs (lower is better).
/// # Safety
/// The arrays hold `count` entries; `out` is valid.
#[no_mangle]
pub unsafe extern "C" fn isl_judge(protocol: u32, min_gain: f64, max_regression: f64, pairs: u32, family_alpha: f64, max_trials: u32,
                                   a: *const f64, b: *const f64, aa: *const f64, correct: *const u8, count: u32, out: *mut IslVerdict) -> i32 {
    let Some(o) = out.as_mut() else { return 1 };
    let rule = Rule { min_gain, max_regression, pairs, family_alpha, max_trials };
    if !(protocol == 1 || protocol == 2) || !rule.valid() || count > 4096 || (count > 0 && (a.is_null() || b.is_null() || aa.is_null() || correct.is_null())) {
        return 1;
    }
    let n = count as usize;
    let triplets: Vec<Triplet> = (0..n).map(|i| Triplet { a: *a.add(i), b: *b.add(i), aa: *aa.add(i), correct: *correct.add(i) == 1 }).collect();
    *o = verdict(protocol, &rule, &triplets);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> IslConfig {
        let mut widths = [0u32; MAX_ACTIONS];
        let mut gather_s = [0.0; MAX_ACTIONS];
        for (i, (w, g)) in [(8, 0.0), (1, 0.0), (2, 0.0), (4, 0.0)].iter().enumerate() {
            widths[i] = *w;
            gather_s[i] = *g;
        }
        IslConfig { abi_version: ABI, action_count: 4, widths, gather_s, max_pending: 8, min_support: 3, ring_capacity: 64, horizon: 5,
                    tau_s: 30.0, alpha_min: 0.05, epsilon: 0.2, synthetic_weight: 0.5, ttft_weight: 1.0, failure_penalty_s: 120.0,
                    reward_scale_s: 10.0, max_rel_std: 1.0, seed: 7 }
    }

    fn st(q: f64) -> IslState {
        let mut s = IslState { queue: q, arrival_rate: 0.5, prompt_chars: 300.0, max_tokens: 64.0, ..Default::default() };
        for k in 0..WAITING {
            let m = (k + 1).min(q.max(1.0) as usize) as f64;
            s.age_cum[k] = 0.2 * m;
            s.chars_cum[k] = 300.0 * m;
        }
        s
    }

    /// Synthetic plant: width w serves min(q, w) in 1 + 0.3 (n - 1) s; first content arrives after
    /// all n prefills (0.1 s each); a narrow width leaves the rest of the queue waiting.
    fn plant(q: f64, w: u32) -> IslTransition {
        let n = q.min(f64::from(w));
        let service = 1.0 + 0.3 * (n - 1.0);
        let first = 0.05 + 0.03 * n; // prefill of every member before first content
        let ttft = 0.2 * n + n * first;
        let next = (q - n).max(0.0);
        IslTransition { state: st(q), next: st(next), action: 0, served: n as u32, ok: n as u32, bad: 0, first_known: 1, duration_s: service,
                        service_s: service, ttft_sum_s: ttft, first_exec_s: first, area_s: q * service, unix_s: 1.0e9 }
    }

    #[test]
    fn invalid_config_and_inputs_are_refused() {
        let mut c = cfg();
        c.widths[0] = 0;
        assert!(Learner::new(c).is_none());
        let mut c = cfg();
        c.widths[5] = 2; // beyond action_count
        assert!(Learner::new(c).is_none());
        let mut l = Learner::new(cfg()).unwrap();
        assert!(l.decide(&st(f64::NAN), 0b1111, false).is_none());
        assert!(l.decide(&st(2.0), 0b1110, false).is_none(), "the reference must stay eligible");
        assert!(l.decide(&st(2.0), 0b1_0000, false).is_none());
        let mut t = plant(2.0, 8);
        t.action = 9;
        assert!(!l.observe(&t));
    }

    #[test]
    fn rejections_cost_more_than_serving_the_request() {
        let l = Learner::new(cfg()).unwrap();
        // The contract sets the penalty above (1 + ttft_weight) x the request deadline: serving a
        // request that waited until just before its 59 s deadline still beats rejecting it.
        assert!(l.reward(59.0, 59.0, 0.0) > l.reward(0.0, 0.0, 1.0));
    }

    #[test]
    fn the_model_predicts_before_it_learns_and_its_error_falls() {
        let mut l = Learner::new(cfg()).unwrap();
        for i in 0..60 {
            let q = 1.0 + (i % 6) as f64;
            let mut t = plant(q, 4);
            t.action = 3;
            assert!(l.observe(&t));
        }
        let status = l.status();
        assert_eq!(status.model_observations, 60);
        assert!(status.mean_abs_rel_error[0] < 0.05, "{:?}", status.mean_abs_rel_error);
        let d = l.decide(&st(3.0), 0b1000 | 1, false).unwrap();
        assert!(d.predicted_service_std > 0.0);
        let p = l.models[3].predict(&l.phi(&st(3.0), 3));
        assert!((p[0].0 - 1.6).abs() < 0.05, "{p:?}");
        assert!(p[0].1.sqrt() < 0.2);
    }

    /// Q-learning bootstraps: with the same immediate reward, an action that leaves a long queue
    /// is worth less than one that empties it, because the next state's value differs.
    #[test]
    fn delayed_effects_reach_the_action_through_the_next_state() {
        let mut l = Learner::new(cfg()).unwrap();
        // State q = 6 is costly (taught directly).
        for _ in 0..50 {
            let mut t = plant(6.0, 8);
            t.area_s = 60.0;
            t.next = st(0.0);
            assert!(l.observe(&t));
        }
        for _ in 0..50 {
            // From q = 3: action 1 leaves 6 waiting, action 3 leaves none, same immediate cost.
            let mut a = plant(3.0, 1);
            a.action = 1;
            a.next = st(6.0);
            a.area_s = 3.0;
            let mut b = plant(3.0, 4);
            b.action = 3;
            b.next = st(0.0);
            b.area_s = 3.0;
            b.ttft_sum_s = a.ttft_sum_s;
            assert!(l.observe(&a) && l.observe(&b));
        }
        let s = l.bin(&st(3.0));
        assert!(l.q[s * MAX_ACTIONS + 3] > l.q[s * MAX_ACTIONS + 1] + 1.0);
        assert_eq!(l.policy()[s], 3);
    }

    /// RSI3-LIVE-P2: summed first content = what is known at the decision (ages, gather) plus
    /// expected members x the model's first-content time; the model learns a per-member slope.
    #[test]
    fn first_content_is_decomposed_into_known_waiting_and_a_per_member_model() {
        let mut l = Learner::new(cfg()).unwrap();
        let before = l.predict(&st(3.0), 3).unwrap();
        assert_eq!(before.supported, 0, "no real observation: the prediction is unusable");
        for i in 0..120 {
            let q = 1.0 + (i % 6) as f64;
            let mut t = plant(q, 4);
            t.action = 3;
            assert!(l.observe(&t));
        }
        for q in [1.0, 2.0, 4.0] {
            let p = l.predict(&st(q), 3).unwrap();
            let n = q.min(4.0);
            assert!((p.expected_members - n).abs() < 1e-12);
            assert!((p.known_wait_s - st(q).age_cum[n as usize - 1]).abs() < 1e-12, "no gather: only ages");
            assert!((p.predicted_first_s - (0.05 + 0.03 * n)).abs() < 0.01, "{p:?}");
            assert!((p.predicted_ttft_sum_s - (p.known_wait_s + n * p.predicted_first_s)).abs() < 1e-12);
        }
        assert_eq!(l.predict(&st(4.0), 3).unwrap().supported, 1);
        assert!(l.status().mean_abs_rel_error[1] < 0.05);
        // A gather wait on a group that is not full adds the wait for each known member.
        let mut c = cfg();
        c.action_count = 5;
        c.widths[4] = 4;
        c.gather_s[4] = 0.1;
        let g = Learner::new(c).unwrap().predict(&st(2.0), 4).unwrap();
        assert!((g.known_wait_s - (st(2.0).age_cum[1] + 2.0 * 0.1 + 0.05 * 0.1 * 0.5)).abs() < 1e-12);
    }

    #[test]
    fn a_group_without_first_content_does_not_train_the_model() {
        let mut l = Learner::new(cfg()).unwrap();
        for _ in 0..40 {
            l.observe(&plant(2.0, 8));
        }
        let before = (l.models[0].clone(), l.status().model_observations, l.status().mean_abs_rel_error);
        let mut cancelled = plant(2.0, 8);
        cancelled.ok = 0;
        cancelled.bad = 2;
        cancelled.first_exec_s = 0.0;
        cancelled.first_known = 0;
        assert!(l.observe(&cancelled));
        assert_eq!(l.models[0], before.0);
        assert_eq!((l.status().model_observations, l.status().mean_abs_rel_error), (before.1, before.2));
        assert_eq!(l.status().real_updates, 41, "its cost still trains the Q-learner");
        // RSI3-LIVE (f): served, but a member's first-content time is missing: missing, not zero.
        let mut unknown = plant(2.0, 8);
        unknown.first_known = 0;
        unknown.first_exec_s = 0.0;
        assert!(l.observe(&unknown));
        assert_eq!(l.models[0], before.0);
        assert_eq!(l.status().model_observations, before.1);
        let mut invalid = plant(2.0, 8);
        invalid.first_known = 2;
        assert!(!l.observe(&invalid));
    }

    #[test]
    fn exploration_is_directed_and_only_when_allowed() {
        let mut c = cfg();
        c.epsilon = 1.0;
        let mut l = Learner::new(c).unwrap();
        let d = l.decide(&st(2.0), 0b1111, false).unwrap();
        assert_eq!((d.action, d.explored), (0, 0));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..4 {
            let d = l.decide(&st(2.0), 0b0101, true).unwrap();
            assert!(d.explored == 1 && [0, 2].contains(&d.action));
            let mut t = plant(2.0, 2);
            t.action = d.action;
            l.observe(&t);
            seen.insert(d.action);
        }
        assert_eq!(seen.len(), 2, "least-visited first covers every eligible action");
    }

    #[test]
    fn dreams_stop_outside_support_and_stay_separate_from_real_experience() {
        let mut l = Learner::new(cfg()).unwrap();
        assert_eq!(l.dream(10, 1_000_000_000).rollouts, 0, "nothing to dream from");
        let mut t = plant(3.0, 8);
        l.observe(&t);
        let r = l.dream(5, 1_000_000_000);
        assert_eq!(r.rollouts, 5);
        assert_eq!(r.stopped_unsupported, 5, "one observation is below min_support");
        assert_eq!(l.status().synthetic_updates, 0);
        for _ in 0..20 {
            for (a, w) in [(0usize, 8u32), (1, 1), (2, 2), (3, 4)] {
                for q in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0] {
                    t = plant(q, w);
                    t.action = a as u32;
                    l.observe(&t);
                }
            }
        }
        let real = l.status().real_updates;
        let ring = l.status().ring_len;
        let r = l.dream(50, 1_000_000_000);
        assert!(r.steps > 0, "{r:?}");
        let s = l.status();
        assert_eq!(s.real_updates, real);
        assert_eq!(s.ring_len, ring, "synthetic transitions are never stored as experience");
        assert_eq!(s.synthetic_updates, u64::from(r.steps));
        assert_eq!(s.depth_hist.iter().sum::<u64>(), s.rollouts, "every rollout has a depth");
        assert_eq!((1..8).map(|d| d as u64 * s.depth_hist[d]).sum::<u64>() <= s.rollout_steps, true);
        assert_eq!(r.rollouts, r.stopped_unsupported + r.stopped_uncertain + r.stopped_idle + r.stopped_horizon + r.budget_exhausted.min(1));
    }

    #[test]
    fn a_zero_budget_dreams_nothing() {
        let mut l = Learner::new(cfg()).unwrap();
        l.observe(&plant(2.0, 8));
        let r = l.dream(1000, 0);
        assert_eq!((r.steps, r.budget_exhausted), (0, 1));
    }

    #[test]
    fn checkpoints_round_trip_exactly_and_refuse_another_config_or_corruption() {
        let mut l = Learner::new(cfg()).unwrap();
        for i in 0..80 {
            let mut t = plant(1.0 + (i % 5) as f64, 4);
            t.action = (i % 4) as u32;
            l.observe(&t);
        }
        l.dream(20, 1_000_000_000);
        let bytes = l.encode();
        assert_eq!(l.encoded_len(), bytes.len());
        let back = Learner::decode(cfg(), &bytes).unwrap();
        assert_eq!(back, l);
        assert_eq!(back.encode(), bytes);
        let mut other = cfg();
        other.ttft_weight = 2.0;
        assert!(Learner::decode(other, &bytes).is_none());
        assert!(Learner::decode(cfg(), &bytes[..bytes.len() - 1]).is_none());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Learner::decode(cfg(), &extra).is_none());
    }

    #[test]
    fn the_ring_is_bounded_and_retention_prunes_old_transitions() {
        let mut l = Learner::new(cfg()).unwrap();
        for i in 0..100 {
            let mut t = plant(2.0, 8);
            t.unix_s = f64::from(i);
            l.observe(&t);
        }
        assert_eq!(l.ring.len(), 64);
        assert_eq!(l.prune(90.0), 54);
        assert_eq!(l.ring.len(), 10);
        assert!(l.ring.iter().all(|t| t.unix_s >= 90.0));
        let back = Learner::decode(cfg(), &l.encode()).unwrap();
        assert_eq!(back, l);
    }

    #[test]
    fn drift_retires_the_model() {
        let mut l = Learner::new(cfg()).unwrap();
        for _ in 0..80 {
            l.observe(&plant(2.0, 8));
        }
        assert_eq!(l.status().drift_resets, 0);
        for _ in 0..80 {
            let mut t = plant(2.0, 8);
            t.service_s *= 5.0;
            t.ttft_sum_s *= 5.0;
            t.first_exec_s *= 5.0;
            l.observe(&t);
        }
        assert!(l.status().drift_resets >= 1);
    }

    #[test]
    fn judge_reuses_the_optimizer_rule() {
        let rule = Rule { min_gain: 0.03, max_regression: 0.10, pairs: 8, family_alpha: 0.05, max_trials: 4 };
        let fast: Vec<Triplet> = (0..8).map(|_| Triplet { a: 10.0, b: 8.0, aa: 10.1, correct: true }).collect();
        assert_eq!(verdict(2, &rule, &fast).status, 1);
        assert_eq!(verdict(2, &rule, &fast[..3]).status, 0);
        let mut wrong = fast.clone();
        wrong[0].correct = false;
        assert_eq!((verdict(2, &rule, &wrong).status, verdict(2, &rule, &wrong).reason), (2, 1));
        let tie: Vec<Triplet> = (0..8).map(|_| Triplet { a: 10.0, b: 10.0, aa: 10.0, correct: true }).collect();
        assert_eq!((verdict(2, &rule, &tie).status, verdict(2, &rule, &tie).reason), (2, 5));
    }
}
