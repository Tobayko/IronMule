//! OPT3 step B: does the background-thread decision have headroom for adaptation at all?
//!
//! Fixed in backlog OPT3 before data: per load level (idle, light, heavy) and thread count, blocks of
//! constant slices in a rotated order, plus blocks that alternate the smallest and largest thread
//! count every slice under light load (switch cost). Per cell: chunks per slice `rho` and the
//! foreground violation share `v`; cost per chunk `c = (1 + penalty v) / rho` (OPT1's slice reward
//! per unit of progress). The oracle picks the cheapest thread count per level, the best fixed
//! setting one count for all levels, both weighted by the schedule chain's stationary level shares;
//! headroom `H = 1 - oracle / fixed`, with a bootstrap interval over slices. `H` is an upper bound for
//! any policy: a real one sees the previous slice's level and pays its switches.
//!
//! Validity (fixed 2026-10-13 before the Kaggle calibration, backlog OPT2-CAL): a constant cell or a switch-block
//! thread count with fewer than half of its slices complete, or a non-finite headroom, interval or switch loss, makes
//! the run invalid, and an invalid run never releases OPT2.

use crate::episode::{schedule, EpisodeConfig, Workers};
use crate::json::Value;

pub const LEVELS: usize = 3;

#[derive(Clone, Debug)]
pub struct SliceRow {
    pub level: u8,
    /// Index into the episode config's thread set; `None` marks a switch-block slice.
    pub action: Option<usize>,
    pub threads: usize,
    pub chunks: usize,
    pub violated: bool,
    pub e2e_p95: Option<f64>,
    pub service_p95: Option<f64>,
    /// The job finished inside this slice: a partial slice, excluded from `rho` and `v`.
    pub partial: bool,
}

impl SliceRow {
    pub fn to_json(&self) -> Value {
        Value::obj().with("level", self.level as u64).with("action", self.action).with("threads", self.threads)
            .with("chunks", self.chunks).with("violated", self.violated).with("e2e_p95_ms", self.e2e_p95)
            .with("service_p95_ms", self.service_p95).with("partial", self.partial)
    }
}

/// The blocks in run order: (level, Some(action)) constant, (1, None) the light-load switch block.
pub fn plan(actions: usize, reps: usize) -> Vec<(u8, Option<usize>)> {
    let cells: Vec<(u8, Option<usize>)> = (0..LEVELS as u8).flat_map(|l| (0..actions).map(move |a| (l, Some(a)))).collect();
    let mut out = Vec::new();
    for r in 0..reps {
        let shift = (r * cells.len()) / reps.max(1);
        let mut order: Vec<_> = (0..cells.len()).map(|i| cells[(i + shift) % cells.len()]).collect();
        if r % 2 == 1 {
            order.reverse();
        }
        out.extend(order);
        out.push((1, None));
    }
    out
}

/// Run the plan for real: `slices` per block; the job restarts when it finishes (its digest checked).
pub fn run(w: &mut Workers, cfg: &EpisodeConfig, targets: &[f64; 3], reference_digest: &str, reps: usize, slices: usize,
           mut on_row: impl FnMut(&SliceRow) -> Result<(), String>) -> Result<Vec<SliceRow>, String> {
    let n = cfg.threads.len();
    let mut rows = Vec::new();
    w.new_job(cfg)?;
    let mut k = 0u64;
    for (b, (level, action)) in plan(n, reps).into_iter().enumerate() {
        for i in 0..slices {
            let a = action.unwrap_or(if i % 2 == 0 { 0 } else { n - 1 });
            let sl = w.slice(cfg, level, cfg.threads[a], 900_000 + k)?;
            k += 1;
            let row = SliceRow { level, action, threads: cfg.threads[a], chunks: sl.chunks,
                                 violated: sl.e2e_p95.is_some_and(|p| p > targets[level as usize]),
                                 e2e_p95: sl.e2e_p95, service_p95: sl.service_p95, partial: sl.finished };
            on_row(&row)?;
            rows.push(row);
            if sl.finished {
                let d = w.finish_job(cfg)?;
                if d != reference_digest {
                    return Err(format!("calibration block {b}: the job's output differs from the reference"));
                }
                w.new_job(cfg)?;
            }
        }
    }
    Ok(rows)
}

/// Stationary shares of the foreground levels under the episode schedule chain.
pub fn level_weights() -> [f64; 3] {
    let s = schedule(1, 100_000);
    let mut w = [0.0; 3];
    for l in s {
        w[l as usize] += 1e-5;
    }
    w
}

struct Cells {
    /// (sum chunks, violations, slices) per (level, action)
    acc: Vec<[(f64, f64, f64); LEVELS]>,
}

impl Cells {
    fn from(rows: &[&SliceRow], actions: usize) -> Cells {
        let mut acc = vec![[(0.0, 0.0, 0.0); LEVELS]; actions];
        for r in rows {
            if let (Some(a), false) = (r.action, r.partial) {
                let c = &mut acc[a][r.level as usize];
                c.0 += r.chunks as f64;
                c.1 += r.violated as u8 as f64;
                c.2 += 1.0;
            }
        }
        Cells { acc }
    }

    /// Cost of `a` at `l` relative to the best fixed setting there: per-level gain of the oracle's choice.
    fn level_gain(&self, l: usize, fixed_a: usize, penalty: f64) -> f64 {
        let best = (0..self.acc.len()).map(|a| self.cost(a, l, penalty)).fold(f64::INFINITY, f64::min);
        1.0 - best / self.cost(fixed_a, l, penalty)
    }

    fn cost(&self, a: usize, l: usize, penalty: f64) -> f64 {
        let (chunks, viol, n) = self.acc[a][l];
        if n == 0.0 || chunks == 0.0 {
            return f64::INFINITY;
        }
        (1.0 + penalty * viol / n) / (chunks / n)
    }

    /// (oracle, best fixed, best fixed action, oracle action per level)
    fn headroom(&self, w: &[f64; 3], penalty: f64) -> (f64, f64, usize, [usize; 3]) {
        let n = self.acc.len();
        let mut oracle_a = [0; 3];
        let mut oracle = 0.0;
        for l in 0..LEVELS {
            let a = (0..n).min_by(|&x, &y| self.cost(x, l, penalty).total_cmp(&self.cost(y, l, penalty))).unwrap();
            oracle_a[l] = a;
            oracle += w[l] * self.cost(a, l, penalty);
        }
        let fixed_of = |a: usize| (0..LEVELS).map(|l| w[l] * self.cost(a, l, penalty)).sum::<f64>();
        let fixed_a = (0..n).min_by(|&x, &y| fixed_of(x).total_cmp(&fixed_of(y))).unwrap();
        (oracle, fixed_of(fixed_a), fixed_a, oracle_a)
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

/// Headroom, its bootstrap interval, the switch cost and the verdict fixed in OPT3.
pub fn analyse(rows: &[SliceRow], cfg: &EpisodeConfig, resamples: usize, seed: u64) -> Value {
    let n = cfg.threads.len();
    let w = level_weights();
    let all: Vec<&SliceRow> = rows.iter().collect();
    let cells = Cells::from(&all, n);
    let (oracle, fixed, fixed_a, oracle_a) = cells.headroom(&w, cfg.penalty);
    let h = 1.0 - oracle / fixed;
    // Bootstrap: resample slices within each (level, action) cell.
    let mut groups: std::collections::BTreeMap<(u8, usize), Vec<&SliceRow>> = Default::default();
    for r in rows.iter().filter(|r| !r.partial) {
        if let Some(a) = r.action {
            groups.entry((r.level, a)).or_default().push(r);
        }
    }
    let mut s = crate::rl::stream(seed);
    let mut next = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    // Per level: how much the level's cheapest setting saves against the best fixed one (descriptive, not the gate).
    let mut level_gains: [Vec<f64>; LEVELS] = Default::default();
    let mut hs: Vec<f64> = (0..resamples).map(|_| {
        let sample: Vec<&SliceRow> = groups.values().flat_map(|g| (0..g.len()).map(|_| g[next(g.len())]).collect::<Vec<_>>()).collect();
        let c = Cells::from(&sample, n);
        let (o, f, fa, _) = c.headroom(&w, cfg.penalty);
        for (l, g) in level_gains.iter_mut().enumerate() {
            g.push(c.level_gain(l, fa, cfg.penalty));
        }
        1.0 - o / f
    }).collect();
    hs.sort_by(f64::total_cmp);
    let ci = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { vec![f64::NAN, f64::NAN] } else { vec![quantile(v, 0.025), quantile(v, 0.975)] }
    };
    let (lo, hi) = if hs.is_empty() { (f64::NAN, f64::NAN) } else { (quantile(&hs, 0.025), quantile(&hs, 0.975)) };
    // Switch cost: alternating smallest/largest thread count under light load vs the mean of the two constants.
    // Per thread count, then averaged: partial slices (dropped) are mostly fast ones, so a pooled mean
    // over the alternating block would weight the slow setting more and invent a switch cost.
    let alt: Vec<&SliceRow> = rows.iter().filter(|r| r.action.is_none() && !r.partial).collect();
    let alt_mean = |t: usize| {
        let v: Vec<f64> = alt.iter().filter(|r| r.threads == t).map(|r| r.chunks as f64).collect();
        if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    let alt_rho = 0.5 * (alt_mean(cfg.threads[0]) + alt_mean(cfg.threads[n - 1]));
    let rho = |a: usize, l: usize| { let c = cells.acc[a][l]; c.0 / c.2.max(1.0) };
    let const_rho = 0.5 * (rho(0, 1) + rho(n - 1, 1));
    let switch_loss = if !alt_rho.is_finite() || const_rho == 0.0 { f64::NAN } else { 1.0 - alt_rho / const_rho };
    let alt_v = alt.iter().filter(|r| r.violated).count() as f64 / alt.len().max(1) as f64;
    let net = h - switch_loss.max(0.0);
    // Validity: every cell and both switch thread counts need at least half of their slices complete.
    let mut invalid = Vec::new();
    for l in 0..LEVELS {
        for a in 0..n {
            let all = rows.iter().filter(|r| r.action == Some(a) && r.level as usize == l).count();
            let done = cells.acc[a][l].2 as usize;
            if all == 0 || 2 * done < all {
                invalid.push(format!("cell level {l} threads {}: {done} of {all} slices complete", cfg.threads[a]));
            }
            if !cells.cost(a, l, cfg.penalty).is_finite() {
                invalid.push(format!("cell level {l} threads {}: no chunks, cost not finite", cfg.threads[a]));
            }
        }
    }
    for t in [cfg.threads[0], cfg.threads[n - 1]] {
        let all = rows.iter().filter(|r| r.action.is_none() && r.threads == t).count();
        let done = alt.iter().filter(|r| r.threads == t).count();
        if all == 0 || 2 * done < all {
            invalid.push(format!("switch block threads {t}: {done} of {all} slices complete"));
        }
    }
    if n < 2 {
        invalid.push(format!("{n} thread count(s): nothing to choose between"));
    }
    for (name, x) in [("headroom", h), ("headroom ci95 low", lo), ("switch loss", switch_loss)] {
        if !x.is_finite() {
            invalid.push(format!("{name} is not finite"));
        }
    }
    let verdict = if !invalid.is_empty() {
        "invalid: missing or incomplete measurements, no release"
    } else if !(net >= 0.05) || !(lo > 0.0) {
        "stop: no headroom for adaptation"
    } else {
        "headroom: OPT2 may run"
    };
    let table: Vec<Value> = (0..LEVELS).flat_map(|l| (0..n).map(move |a| (l, a))).map(|(l, a)| {
        let in_cell: Vec<&SliceRow> = rows.iter().filter(|r| r.action == Some(a) && r.level as usize == l && !r.partial).collect();
        let med = |f: &dyn Fn(&SliceRow) -> Option<f64>| {
            let mut v: Vec<f64> = in_cell.iter().filter_map(|r| f(r)).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() { None } else { Some(quantile(&v, 0.5)) }
        };
        let c = cells.acc[a][l];
        Value::obj().with("level", l).with("threads", cfg.threads[a]).with("slices", c.2).with("rho_chunks_per_slice", rho(a, l))
            .with("violation_share", c.1 / c.2.max(1.0)).with("cost_per_chunk", cells.cost(a, l, cfg.penalty))
            .with("median_e2e_p95_ms", med(&|r| r.e2e_p95)).with("median_service_p95_ms", med(&|r| r.service_p95))
    }).collect();
    Value::obj().with("cells", table).with("level_weights", w.to_vec())
        .with("oracle_cost", oracle).with("oracle_threads", oracle_a.iter().map(|&a| cfg.threads[a]).collect::<Vec<_>>())
        .with("best_fixed_cost", fixed).with("best_fixed_threads", cfg.threads[fixed_a])
        .with("headroom", h).with("headroom_ci95", vec![lo, hi]).with("resamples", resamples)
        .with("switch_alternating_rho", alt_rho).with("switch_constant_mean_rho", const_rho).with("switch_loss", switch_loss)
        .with("switch_violation_share", alt_v).with("net_headroom", net).with("partial_slices", rows.iter().filter(|r| r.partial).count())
        .with("level_gain", (0..LEVELS).map(|l| cells.level_gain(l, fixed_a, cfg.penalty)).collect::<Vec<_>>())
        .with("level_gain_ci95", level_gains.iter_mut().map(|g| Value::from(ci(g))).collect::<Vec<_>>())
        .with("invalid", invalid).with("verdict", verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EpisodeConfig {
        EpisodeConfig { job_chunks: 100, chunk_size: 1, slice_ms: 500, max_slices: 60, rates: [0.0, 20.0, 60.0], fg_size: 1,
                        penalty: 4.0, threads: vec![1, 4] }
    }

    fn rows(rho: [[usize; 3]; 2], viol: [[bool; 3]; 2]) -> Vec<SliceRow> {
        let mut out = Vec::new();
        for a in 0..2 {
            for l in 0..3 {
                for k in 0..10 {
                    // a little spread so the bootstrap has something to resample
                    out.push(SliceRow { level: l as u8, action: Some(a), threads: [1, 4][a], chunks: rho[a][l] + k % 2,
                                        violated: viol[a][l], e2e_p95: Some(1.0), service_p95: Some(1.0), partial: false });
                }
            }
        }
        for k in 0..10 {
            // alternating at the constant blocks' rates: no switch cost
            out.push(SliceRow { level: 1, action: None, threads: [1, 4][k % 2], chunks: [rho[0][1], rho[1][1]][k % 2], violated: false,
                                e2e_p95: None, service_p95: None, partial: false });
        }
        out
    }

    #[test]
    fn headroom_exists_only_when_levels_want_different_settings() {
        // 4 threads always faster and never violating: one fixed setting is the oracle, no headroom
        let same = analyse(&rows([[10, 10, 10], [40, 40, 40]], [[false; 3]; 2]), &cfg(), 200, 1);
        assert_eq!(same.num("headroom"), Some(0.0));
        assert_eq!(same.str("verdict"), Some("stop: no headroom for adaptation"));
        // 4 threads violate under heavy load: heavy wants 1 thread, the others 4
        let differ = analyse(&rows([[10, 10, 10], [40, 40, 40]], [[false; 3], [false, false, true]]), &cfg(), 200, 1);
        assert!(differ.num("headroom").unwrap() > 0.1, "{}", differ.dump());
        assert_eq!(differ.arr("oracle_threads").unwrap().iter().filter_map(|x| x.as_f64()).collect::<Vec<_>>(), vec![4.0, 4.0, 1.0]);
        assert_eq!(differ.str("verdict"), Some("headroom: OPT2 may run"));
        let w = level_weights();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6 && w.iter().all(|x| *x > 0.25));
        // a partial slice (job finished inside it) never counts
        let mut r = rows([[10, 10, 10], [40, 40, 40]], [[false; 3]; 2]);
        r.push(SliceRow { level: 0, action: Some(0), threads: 1, chunks: 0, violated: true, e2e_p95: None, service_p95: None, partial: true });
        assert_eq!(analyse(&r, &cfg(), 10, 1).num("headroom"), Some(0.0));
        // fast alternating slices lost as partial must not look like a switch cost (pooled mean would say 30 %)
        let mut r = rows([[10, 10, 10], [40, 40, 40]], [[false; 3]; 2]);
        for x in r.iter_mut().filter(|x| x.action.is_none() && x.threads == 4).take(4) {
            x.partial = true;
        }
        let loss = analyse(&r, &cfg(), 10, 1).num("switch_loss").unwrap();
        assert!(loss.abs() < 0.03, "switch loss {loss}");
        // OPT2-CAL validity: a cell with most slices partial, or a switch block without one of its thread counts,
        // can never release OPT2, even where the remaining slices would show headroom
        let differ_rows = || rows([[10, 10, 10], [40, 40, 40]], [[false; 3], [false, false, true]]);
        let mut r = differ_rows();
        for x in r.iter_mut().filter(|x| x.action == Some(1) && x.level == 0).take(6) {
            x.partial = true;
        }
        let a = analyse(&r, &cfg(), 50, 1);
        assert!(a.str("verdict").unwrap().starts_with("invalid"), "{}", a.dump());
        let mut r = differ_rows();
        r.retain(|x| !(x.action.is_none() && x.threads == 4));
        let a = analyse(&r, &cfg(), 50, 1);
        assert!(a.str("verdict").unwrap().starts_with("invalid"), "{}", a.dump());
        let mut r = differ_rows();
        for x in r.iter_mut().filter(|x| x.action == Some(1) && x.level == 0).take(5) {
            x.partial = true; // exactly half complete still counts
        }
        assert_eq!(analyse(&r, &cfg(), 50, 1).str("verdict"), Some("headroom: OPT2 may run"));
        // complete slices without progress make a cell's cost infinite, which would steer fixed against oracle
        let mut r = differ_rows();
        for x in r.iter_mut().filter(|x| x.action == Some(1) && x.level == 0) {
            x.chunks = 0;
        }
        let a = analyse(&r, &cfg(), 50, 1);
        assert!(a.str("verdict").unwrap().starts_with("invalid"), "{}", a.dump());
        // per-level gain: heavy wants 1 thread and saves there, idle and light none
        let g: Vec<f64> = analyse(&differ_rows(), &cfg(), 50, 1).arr("level_gain").unwrap().iter().filter_map(|x| x.as_f64()).collect();
        assert!(g[0].abs() < 1e-9 && g[1].abs() < 1e-9 && g[2] > 0.2, "{g:?}");
        // the plan rotates, alternates direction and ends every repetition with a switch block
        let p = plan(2, 3);
        assert_eq!(p.len(), 3 * 7);
        assert_eq!(p.iter().filter(|b| b.1.is_none()).count(), 3);
        assert_ne!(p[0], p[7]);
    }
}
