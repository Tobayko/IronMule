//! OPT1's equal-budget comparison for the sequential decision, as fixed in the backlog before data:
//! five strategies get the same wall-clock budget of real interaction (episodes, learning, replay,
//! switching all counted by one clock), then each frozen policy runs the same holdout schedules,
//! which no strategy saw; the two simple strategies run twice per schedule (A/A control).
//! Every episode's job output must equal the reference digest or the strategy is invalid.
//! Every slice of every episode is journaled as a discovery-tree node (predecessor = previous slice).

use crate::dream::{self, ReplayModel};
use crate::episode::{self, EpisodeConfig, EpisodeResult, Policy, Transition, Workers};
use crate::json::Value;
use crate::qualify::sign_test_p;
use crate::rl::{self, QTable};
use crate::store::Journal;
use ironmule_experiment_planner::{Planner, PlannerConfig};
use std::time::Instant;

pub struct Setup<'a> {
    pub cfg: EpisodeConfig,
    pub targets: [f64; 3],
    pub reference_digest: String,
    pub budget_s: f64,
    pub holdout: Vec<u64>,
    pub train_seed0: u64,
    pub identity: String,
    pub journal: &'a mut Journal,
    pub log: &'a mut dyn FnMut(&str),
}

impl Setup<'_> {
    fn record_episode(&mut self, strategy: &str, phase: &str, r: &EpisodeResult) -> Result<(), String> {
        if r.digest != self.reference_digest {
            return Err(format!("{strategy}: episode {} produced a different job output", r.seed));
        }
        let mut prev = Value::Null;
        for t in &r.transitions {
            let rec = self.journal.append(&self.identity, "tree_node", t.to_json().with("strategy", strategy).with("phase", phase)
                .with("episode_seed", r.seed).with("predecessor", prev.clone()))?;
            prev = rec.get("hash").cloned().unwrap_or(Value::Null);
        }
        self.journal.append(&self.identity, "episode", r.to_json(false).with("strategy", strategy).with("phase", phase))?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Frozen {
    Fixed(usize),
    LoadRule,
    Q(QTable),
}

impl Frozen {
    fn policy(&mut self) -> Policy<'_> {
        match self {
            Frozen::Fixed(a) => Policy::Fixed(*a),
            Frozen::LoadRule => Policy::LoadRule,
            Frozen::Q(t) => Policy::Q { table: t, explore: false },
        }
    }

    pub fn describe(&self, cfg: &EpisodeConfig) -> Value {
        match self {
            Frozen::Fixed(a) => Value::obj().with("kind", "fixed").with("threads", cfg.threads[*a]),
            Frozen::LoadRule => Value::obj().with("kind", "load_rule"),
            Frozen::Q(t) => Value::obj().with("kind", "q_greedy").with("table", t.to_json()),
        }
    }
}

pub struct Trained {
    pub name: String,
    pub policy: Frozen,
    pub episodes: usize,
    pub spent_s: f64,
    pub notes: Value,
}

fn train_seed(setup: &Setup, k: usize) -> u64 {
    setup.train_seed0 + k as u64
}

/// S1: sweep fixed thread counts round-robin over training schedules; keep the best mean return.
pub fn sweep(setup: &mut Setup, w: &mut Workers) -> Result<Trained, String> {
    let t0 = Instant::now();
    let n = setup.cfg.threads.len();
    let mut sums = vec![(0.0, 0usize); n];
    let mut k = 0;
    while t0.elapsed().as_secs_f64() < setup.budget_s {
        let a = k % n;
        let r = episode::run_episode(w, &setup.cfg, &setup.targets, train_seed(setup, k), &mut Policy::Fixed(a))?;
        setup.record_episode("sweep", "train", &r)?;
        sums[a].0 += r.ret;
        sums[a].1 += 1;
        k += 1;
    }
    let means: Vec<f64> = sums.iter().map(|(s, c)| if *c > 0 { s / *c as f64 } else { f64::NEG_INFINITY }).collect();
    let best = (0..n).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap().then(b.cmp(&a))).unwrap();
    Ok(Trained { name: "sweep_best_fixed".into(), policy: Frozen::Fixed(best), episodes: k, spent_s: t0.elapsed().as_secs_f64(),
                 notes: Value::obj().with("mean_return_per_action", means).with("chosen_threads", setup.cfg.threads[best]) })
}

/// S3: the existing RSI1 planner over fixed settings; one test = one episode; effect = log(cost / cost of
/// the first setting), cost = -return. Stops at the budget or when no test is worth its time.
pub fn planner(setup: &mut Setup, w: &mut Workers) -> Result<Trained, String> {
    let t0 = Instant::now();
    let n = setup.cfg.threads.len();
    let mut p = Planner::new(PlannerConfig { prior_sd: 0.3, noise_sd: 0.1, min_gain_per_s: 0.0, default_cost_s: 5.0 })
        .ok_or("planner config")?;
    let r0 = episode::run_episode(w, &setup.cfg, &setup.targets, train_seed(setup, 0), &mut Policy::Fixed(0))?;
    setup.record_episode("planner", "train", &r0)?;
    let ref_cost = -r0.ret;
    let mut k = 1;
    let mut stop = "budget";
    while t0.elapsed().as_secs_f64() < setup.budget_s {
        let arms: Vec<u32> = (1..n as u32).collect();
        let best = arms.iter().map(|&a| p.status(a as usize)).filter(|s| s.observations > 0)
            .map(|s| s.posterior_mean).fold(0.0f64, f64::min);
        let Some((i, _)) = p.choose(&arms, &vec![0.0; arms.len()], best) else {
            stop = "no test worth its time";
            break;
        };
        let a = arms[i] as usize;
        let r = episode::run_episode(w, &setup.cfg, &setup.targets, train_seed(setup, k), &mut Policy::Fixed(a))?;
        setup.record_episode("planner", "train", &r)?;
        p.observe(a, (-r.ret / ref_cost).ln(), r.wall_s);
        k += 1;
    }
    let mut best = (0usize, 0.0f64);
    for a in 1..n {
        let s = p.status(a);
        if s.observations > 0 && s.posterior_mean < best.1 {
            best = (a, s.posterior_mean);
        }
    }
    Ok(Trained { name: "planner".into(), policy: Frozen::Fixed(best.0), episodes: k, spent_s: t0.elapsed().as_secs_f64(),
                 notes: Value::obj().with("stop", stop).with("chosen_threads", setup.cfg.threads[best.0]).with("posterior_mean", best.1) })
}

fn q_online(setup: &mut Setup, w: &mut Workers, table: &mut QTable, name: &str, t0: Instant, until_s: f64, k0: usize)
            -> Result<(usize, Vec<Vec<Transition>>), String> {
    let mut k = k0;
    let mut trees = Vec::new();
    while t0.elapsed().as_secs_f64() < until_s {
        let r = episode::run_episode(w, &setup.cfg, &setup.targets, train_seed(setup, k), &mut Policy::Q { table, explore: true })?;
        setup.record_episode(name, "train", &r)?;
        trees.push(r.transitions);
        k += 1;
    }
    Ok((k, trees))
}

/// S4: plain Q-learning with the default settings for the whole budget.
pub fn qlearn(setup: &mut Setup, w: &mut Workers) -> Result<Trained, String> {
    let t0 = Instant::now();
    let mut table = QTable::new(rl::DEFAULT, setup.cfg.threads.len(), 4242);
    let budget = setup.budget_s;
    let (k, _) = q_online(setup, w, &mut table, "qlearn", t0, budget, 0)?;
    Ok(Trained { name: "qlearn".into(), notes: Value::obj().with("config", rl::DEFAULT.to_json()).with("updates", table.updates),
                 policy: Frozen::Q(table), episodes: k, spent_s: t0.elapsed().as_secs_f64() })
}

/// S5: Q-learning with the dream meta loop: half the budget online with the default settings, then
/// choose settings by replay (development trees: even seeds; validation trees: odd seeds), relearn
/// from every recorded real transition with them, and spend the rest online. The clock runs throughout.
pub fn qlearn_dream(setup: &mut Setup, w: &mut Workers) -> Result<Trained, String> {
    let t0 = Instant::now();
    let actions = setup.cfg.threads.len();
    let mut first = QTable::new(rl::DEFAULT, actions, 4242);
    let half = setup.budget_s / 2.0;
    let (k, trees) = q_online(setup, w, &mut first, "qlearn_dream", t0, half, 0)?;
    let d0 = Instant::now();
    let dev = ReplayModel::build(trees.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, t)| &t[..]));
    let val = ReplayModel::build(trees.iter().enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, t)| &t[..]));
    let (chosen, report) = dream::select(&dev, &val, actions, 60, 20, &[1, 2, 3]);
    let dream_s = d0.elapsed().as_secs_f64();
    setup.journal.append(&setup.identity, "dream_select", report.clone().with("dream_seconds", dream_s)
        .with("dev_transitions", dev.transitions).with("val_transitions", val.transitions))?;
    // No supported candidate: keep the settings that produced the trees.
    let chosen = chosen.unwrap_or(rl::DEFAULT);
    (setup.log)(&format!("dream chose {} in {dream_s:.2} s", chosen.to_json().dump()));
    // Relearn from the recorded real transitions (one pass, recorded order) with the chosen settings.
    let mut table = QTable::new(chosen, actions, 4343);
    for tree in &trees {
        for t in tree {
            table.update(t.state, t.action, t.reward, t.next);
        }
        table.end_episode();
    }
    let budget = setup.budget_s;
    let (k2, _) = q_online(setup, w, &mut table, "qlearn_dream", t0, budget, k)?;
    Ok(Trained { name: "qlearn_dream".into(), notes: report.with("dream_seconds", dream_s).with("online_episodes_before_dream", k),
                 policy: Frozen::Q(table), episodes: k2, spent_s: t0.elapsed().as_secs_f64() })
}

/// Holdout: every frozen policy on every holdout schedule (order rotated per schedule), the two
/// simple strategies twice (A/A). Returns per-strategy returns, violations and wall times.
pub fn holdout(setup: &mut Setup, w: &mut Workers, mut policies: Vec<(String, Frozen)>) -> Result<Value, String> {
    let seeds = setup.holdout.clone();
    let mut runs: Vec<(String, usize)> = Vec::new();
    for (i, (name, _)) in policies.iter().enumerate() {
        runs.push((name.clone(), i));
        if name == "sweep_best_fixed" || name == "load_rule" {
            runs.push((format!("{name}#AA"), i));
        }
    }
    let mut table: std::collections::BTreeMap<String, Vec<Value>> = Default::default();
    for (j, &seed) in seeds.iter().enumerate() {
        for k in 0..runs.len() {
            let (label, i) = runs[(k + j) % runs.len()].clone();
            let r = episode::run_episode(w, &setup.cfg, &setup.targets, seed, &mut policies[i].1.policy())?;
            setup.record_episode(&label, "holdout", &r)?;
            table.entry(label).or_default().push(Value::obj().with("seed", seed).with("return", r.ret)
                .with("violations", r.violations).with("slices", r.slices).with("wall_s", r.wall_s));
        }
        (setup.log)(&format!("holdout schedule {}/{} done", j + 1, seeds.len()));
    }
    Ok(Value::Obj(table.into_iter().map(|(k, v)| (k, Value::Arr(v))).collect()))
}

fn returns(h: &Value, name: &str) -> Vec<f64> {
    h.arr(name).map(|a| a.iter().filter_map(|x| x.num("return")).collect()).unwrap_or_default()
}

fn violations(h: &Value, name: &str) -> f64 {
    h.arr(name).map(|a| a.iter().filter_map(|x| x.num("violations")).sum()).unwrap_or(f64::NAN)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() { f64::NAN } else if s.len() % 2 == 1 { s[s.len() / 2] } else { 0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2]) }
}

/// The fixed decision rule of OPT1: candidate beats base if its median paired return gain is at
/// least 5 % of |base median|, it wins at least 15 of 20 schedules (3/4, rounded up, for other holdout
/// sizes), it violates the foreground
/// target no more often, and the gain exceeds the A/A spread; inside the A/A spread it is a tie.
pub fn decide(h: &Value, candidate: &str, base: &str, aa_spread: f64) -> Value {
    let (c, b) = (returns(h, candidate), returns(h, base));
    let diffs: Vec<f64> = c.iter().zip(&b).map(|(x, y)| x - y).collect();
    let gain = median(&diffs);
    let wins = diffs.iter().filter(|d| **d > 0.0).count() as u32;
    let need = 0.05 * median(&b).abs();
    let (vc, vb) = (violations(h, candidate), violations(h, base));
    let verdict = if c.len() != b.len() || c.is_empty() {
        "invalid"
    } else if gain.abs() <= aa_spread {
        "tie (inside the A/A spread)"
    } else if gain >= need && wins >= (3 * diffs.len() as u32).div_ceil(4) && vc <= vb {
        "accepted"
    } else {
        "rejected"
    };
    Value::obj().with("candidate", candidate).with("base", base).with("median_paired_gain", gain).with("required_gain", need)
        .with("wins", wins as u64).with("schedules", diffs.len()).with("sign_test_p", sign_test_p(wins, diffs.len() as u32))
        .with("violations_candidate", vc).with("violations_base", vb).with("aa_spread", aa_spread).with("verdict", verdict)
}

pub fn analyse(h: &Value) -> Value {
    let names = ["sweep_best_fixed", "load_rule", "planner", "qlearn", "qlearn_dream"];
    let summary: Vec<Value> = names.iter().map(|n| {
        let r = returns(h, n);
        Value::obj().with("strategy", *n).with("median_return", median(&r)).with("violations", violations(h, n))
    }).collect();
    // The best simple strategy: the higher holdout median of the two simple ones (conservative for RL).
    let best_simple = if median(&returns(h, "sweep_best_fixed")) >= median(&returns(h, "load_rule")) { "sweep_best_fixed" } else { "load_rule" };
    let aa: Vec<f64> = returns(h, best_simple).iter().zip(returns(h, &format!("{best_simple}#AA"))).map(|(a, b)| (a - b).abs()).collect();
    let aa_spread = median(&aa);
    Value::obj().with("summary", summary).with("best_simple", best_simple).with("aa_median_abs_diff", aa_spread)
        .with("rl_vs_best_simple", decide(h, "qlearn", best_simple, aa_spread))
        .with("dream_vs_rl", decide(h, "qlearn_dream", "qlearn", aa_spread))
        .with("planner_vs_best_simple", decide(h, "planner", best_simple, aa_spread))
        .with("dream_vs_best_simple", decide(h, "qlearn_dream", best_simple, aa_spread))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(rows: &[(&str, Vec<f64>, f64)]) -> Value {
        let mut v = Value::obj();
        for (name, rets, viol) in rows {
            v.set(name, Value::Arr(rets.iter().map(|r| Value::obj().with("return", *r).with("violations", *viol)).collect()));
        }
        v
    }

    #[test]
    fn the_decision_rule_needs_size_wins_safety_and_more_than_noise() {
        let base: Vec<f64> = (0..20).map(|i| -20.0 - (i % 3) as f64).collect();
        let better: Vec<f64> = base.iter().map(|b| b + 2.0).collect();
        let tiny: Vec<f64> = base.iter().map(|b| b + 0.5).collect();
        let x = h(&[("c", better.clone(), 1.0), ("b", base.clone(), 1.0), ("t", tiny, 1.0), ("unsafe", better, 5.0)]);
        assert_eq!(decide(&x, "c", "b", 0.5).str("verdict"), Some("accepted"));
        assert_eq!(decide(&x, "t", "b", 0.2).str("verdict"), Some("rejected")); // 0.5 < 5 % of 20
        assert_eq!(decide(&x, "c", "b", 3.0).str("verdict"), Some("tie (inside the A/A spread)"));
        assert_eq!(decide(&x, "unsafe", "b", 0.5).str("verdict"), Some("rejected"));
        assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    }
}
