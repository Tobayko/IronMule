//! OPT3: a bounded, resumable learning campaign for the background-thread decision.
//!
//! Rounds of: real training episodes (Q-learning, online) -> a holdout check against a fixed baseline
//! on schedules no learner saw -> a fixed decision -> replay selection of the next round's learner
//! settings from the training trees only. Everything is rebuilt from the journal, so a restart picks
//! up the same experience and state: the learned table is the replay of every completed training
//! episode's recorded updates (verified against the digest the online learner journaled), each
//! attempt has its own journaled RNG stream and schedule seed, and the plan, targets and reference
//! are fixed in the first record.
//!
//! Abort semantics: an attempt journaled without its episode was interrupted. On restart it becomes
//! `attempt_aborted`: it is never counted as a measurement, its updates (held only in memory) are
//! gone, a training attempt keeps a budget charge of the episode cap, and a holdout attempt burns its
//! whole schedule (its remaining arms never run, so no schedule is retried after a look). Activation
//! is journaled as intent then done, and never repeated.
//!
//! Modes: `compare` stays the fresh research comparison; a campaign learns across rounds and only
//! activates a policy its own fixed holdout rule accepted; `managed` (product) runs only an activated
//! policy, else the reference.

use crate::dream::{self, ReplayModel};
use crate::episode::{self, EpisodeConfig, Policy, Transition, Workers};
use crate::experiment::Frozen;
use crate::json::Value;
use crate::qualify::sign_test_p;
use crate::rl::{self, QConfig, QTable};
use crate::sha256;
use crate::store::{self, Journal};
use std::path::Path;
use std::time::Instant;

pub const CONCURRENCY_SCHEMA: &str = "imopt.concurrency_policy.v1";
pub const TRAIN_SEED0: u64 = 10_000;
const ARMS_EVEN: [&str; 4] = ["F", "L", "L'", "F'"];
const ARMS_ODD: [&str; 4] = ["L", "F", "F'", "L'"];

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub rounds: usize,
    /// Real training seconds per round (aborted attempts and replay selection are charged to it).
    pub budget_s: f64,
    /// Holdout schedules per round.
    pub holdout: usize,
    /// The fixed baseline, an index into the thread set.
    pub baseline: usize,
    pub seed: u64,
    pub dream: bool,
    /// OPT2: the learner trains on the dense reward (the holdout still judges the sparse return).
    pub dense: bool,
    /// OPT2: round 1 begins with a sweep of fixed actions (a third of its budget, every action at least
    /// once); the learner then starts with the sweep's best action greedy everywhere.
    pub warm: bool,
    /// Holdout schedule seeds `seed0 + j` (j = 1..holdout) instead of OPT3's `1_000_000 x round + j`.
    pub holdout_seed0: Option<u64>,
}

impl Plan {
    pub fn from_json(v: &Value) -> Option<Plan> {
        let flag = |k: &str, off: &str, on: &str| match v.str(k) { None => Some(false), Some(x) if x == off => Some(false),
                                                                    Some(x) if x == on => Some(true), _ => None };
        Some(Plan { rounds: v.num("rounds")? as usize, budget_s: v.num("budget_s")?, holdout: v.num("holdout")? as usize,
                    baseline: v.num("baseline_action")? as usize, seed: v.str("seed")?.parse().ok()?, dream: v.boolean("dream")?,
                    dense: flag("reward", "sparse", "dense")?, warm: flag("start", "cold", "warm")?,
                    holdout_seed0: match v.str("holdout_seed0") { Some(x) => Some(x.parse().ok()?), None => None } })
    }

    /// OPT3's fields only, unless an OPT2 option is set: a stored OPT3 plan still compares equal.
    pub fn to_json(&self) -> Value {
        let v = Value::obj().with("rounds", self.rounds).with("budget_s", self.budget_s).with("holdout", self.holdout)
            .with("baseline_action", self.baseline).with("seed", format!("{}", self.seed)).with("dream", self.dream);
        let v = if self.dense { v.with("reward", "dense") } else { v };
        let v = if self.warm { v.with("start", "warm") } else { v };
        match self.holdout_seed0 { Some(s0) => v.with("holdout_seed0", format!("{s0}")), None => v }
    }
}

pub fn holdout_seed(round: usize, j: usize) -> u64 {
    1_000_000 * round as u64 + j as u64
}

/// Each attempt's own exploration stream: never shared, never reused after an abort.
pub fn attempt_stream(seed: u64, attempt: u64) -> u64 {
    rl::stream(seed ^ rl::stream(attempt))
}

/// Digest of what was learned (table, counts, settings), not of the exploration generator.
pub fn learned_digest(t: &QTable) -> String {
    let mut v = t.to_json();
    if let Value::Obj(m) = &mut v {
        m.remove("rng");
    }
    sha256::hex_digest(v.dump().as_bytes())
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => 0.5 * (s[n / 2 - 1] + s[n / 2]),
    }
}

pub struct Campaign<'a> {
    pub journal: &'a mut Journal,
    pub id: String,
    pub cfg: EpisodeConfig,
    pub plan: Plan,
    pub targets: [f64; 3],
    pub reference_digest: String,
    pub log: &'a mut dyn FnMut(&str),
}

impl Campaign<'_> {
    fn mine<'b>(&'b self, kind: &'b str) -> impl Iterator<Item = &'b Value> + 'b {
        let id = self.id.clone();
        self.journal.of_kind(kind).filter(move |r| r.str("identity") == Some(id.as_str())).filter_map(|r| r.get("body"))
    }

    fn attempts(&self) -> u64 {
        self.mine("attempt").count() as u64
    }

    fn aborted_charge(&self) -> f64 {
        self.cfg.max_slices as f64 * self.cfg.slice_ms as f64 / 1000.0
    }

    fn round_of(b: &Value) -> usize {
        b.num("round").unwrap_or(0.0) as usize
    }

    /// Mark every interrupted attempt; returns how many were found.
    fn recover(&mut self) -> Result<usize, String> {
        let closed: std::collections::BTreeSet<u64> = self.mine("episode").chain(self.mine("attempt_aborted")).chain(self.mine("attempt_failed"))
            .filter_map(|b| b.num("attempt")).map(|a| a as u64).collect();
        let open: Vec<Value> = self.mine("attempt").filter(|b| b.num("attempt").is_some_and(|a| !closed.contains(&(a as u64))))
            .cloned().collect();
        for b in &open {
            let charged = if matches!(b.str("phase"), Some("train" | "sweep")) { self.aborted_charge() } else { 0.0 };
            self.journal.append(&self.id, "attempt_aborted", b.clone().with("charged_s", charged)
                .with("note", "interrupted: not a measurement; its in-memory updates are discarded"))?;
            (self.log)(&format!("attempt {} ({}) was interrupted: recorded as aborted", b.num("attempt").unwrap_or(-1.0),
                                b.str("phase").unwrap_or("?")));
        }
        Ok(open.len())
    }

    /// Seconds of `phases` charged to `round`: completed and aborted attempts.
    fn charged(&self, round: usize, phases: &[&str]) -> f64 {
        let of = |b: &&Value| b.str("phase").is_some_and(|p| phases.contains(&p)) && Self::round_of(b) == round;
        let ep: f64 = self.mine("episode").filter(of).filter_map(|b| b.num("spent_s")).sum();
        let ab: f64 = self.mine("attempt_aborted").filter(of).filter_map(|b| b.num("charged_s")).sum();
        ep + ab
    }

    /// Training seconds charged to `round`: completed and aborted attempts (a warm start's sweep included),
    /// and the replay selection before it.
    pub fn spent(&self, round: usize) -> f64 {
        let dr: f64 = self.mine("select").filter(|b| Self::round_of(b) + 1 == round).filter_map(|b| b.num("dream_s")).sum();
        self.charged(round, &["train", "sweep"]) + dr
    }

    /// OPT2 warm start: the mean sparse return per fixed action over this campaign's completed sweep
    /// episodes (the learner's initial values; its greedy choice is the best, ties to fewer threads); None
    /// without a warm plan or before every action has one.
    pub fn warm_values(&self) -> Option<Vec<f64>> {
        let n = self.cfg.threads.len();
        let mut sums = vec![(0.0, 0usize); n];
        for b in self.mine("episode").filter(|b| b.str("phase") == Some("sweep")) {
            if let (Some(a), Some(r)) = (b.num("action"), b.num("return")) {
                if (a as usize) < n {
                    sums[a as usize].0 += r;
                    sums[a as usize].1 += 1;
                }
            }
        }
        if !self.plan.warm || sums.iter().any(|s| s.1 == 0) {
            return None;
        }
        Some(sums.iter().map(|(s, c)| s / *c as f64).collect())
    }

    fn sweep_done(&self) -> bool {
        self.charged(1, &["sweep"]) >= self.plan.budget_s / 3.0 && self.warm_values().is_some()
    }

    fn sweep_attempt(&mut self, w: &mut Workers) -> Result<(), String> {
        let n = self.attempts();
        let k = self.mine("attempt").filter(|b| b.str("phase") == Some("sweep")).count();
        let action = k % self.cfg.threads.len();
        let seed = TRAIN_SEED0 + n;
        self.journal.append(&self.id, "attempt", Value::obj().with("attempt", n).with("round", 1usize).with("phase", "sweep")
            .with("seed", seed).with("action", action))?;
        self.fault(n);
        let t0 = Instant::now();
        let r = episode::run_episode(w, &self.cfg, &self.targets, seed, &mut Policy::Fixed(action))?;
        if r.digest != self.reference_digest {
            return self.failed(n, "sweep", &r.digest);
        }
        self.journal.append(&self.id, "episode", r.to_json(false).with("attempt", n).with("round", 1usize).with("phase", "sweep")
            .with("action", action).with("spent_s", t0.elapsed().as_secs_f64()))?;
        Ok(())
    }

    fn holdout_seed_of(&self, round: usize, j: usize) -> u64 {
        self.plan.holdout_seed0.map_or_else(|| holdout_seed(round, j), |s0| s0 + j as u64)
    }

    /// Learner settings for `round`: the latest supported replay choice before it, else the default.
    pub fn config_for(&self, round: usize) -> QConfig {
        self.mine("select").filter(|b| Self::round_of(b) < round).filter_map(|b| b.get("chosen").and_then(QConfig::from_json))
            .last().unwrap_or(rl::DEFAULT)
    }

    /// Training episodes of this campaign (each carries its attempt; records of any other writer never count).
    fn train_trees(&self) -> Vec<(u64, Vec<Transition>)> {
        self.mine("episode").filter(|b| b.str("phase") == Some("train") && b.num("attempt").is_some()).map(|b| {
            let t = b.arr("transitions").map(|a| a.iter().filter_map(Transition::from_json).collect()).unwrap_or_default();
            (b.num("seed").unwrap_or(0.0) as u64, t)
        }).collect()
    }

    /// The learned state: `config` replayed over every completed training episode's real updates, in
    /// journal order. Holdout episodes never enter it.
    pub fn rebuild(&self, config: QConfig) -> QTable {
        let mut table = QTable::new(config, self.cfg.threads.len(), self.plan.seed);
        table.dense = self.plan.dense;
        table.prior = self.warm_values();
        for (_, tree) in self.train_trees() {
            for t in &tree {
                table.update(t.state, t.action, t.reward, t.next);
            }
            table.end_episode();
        }
        table
    }

    /// The online learner journaled its digest after each episode: a rebuild under the same settings must match.
    fn verify(&self, table: &QTable, round: usize) -> Result<(), String> {
        let last = self.mine("episode").filter(|b| b.str("phase") == Some("train") && b.num("attempt").is_some()).last();
        match last {
            Some(b) if Self::round_of(b) == round && b.str("learned_sha256") != Some(learned_digest(table).as_str()) =>
                Err("the rebuilt learner differs from the one journaled online: refusing to continue".into()),
            _ => Ok(()),
        }
    }

    /// Test-only fault injection (`IMOPT_FAULT`, as in `imopt-cpujob`).
    fn has_fault(name: &str) -> bool {
        std::env::var("IMOPT_FAULT").ok().is_some_and(|f| f.split(',').any(|x| x == name))
    }

    fn fault(&self, attempt: u64) {
        if Self::has_fault(&format!("campaign_abort:{attempt}")) {
            std::process::exit(9);
        }
    }

    fn train_attempt(&mut self, w: &mut Workers, round: usize, table: &mut QTable) -> Result<(), String> {
        let n = self.attempts();
        let seed = TRAIN_SEED0 + n;
        table.reseed(attempt_stream(self.plan.seed, n));
        self.journal.append(&self.id, "attempt", Value::obj().with("attempt", n).with("round", round).with("phase", "train")
            .with("seed", seed).with("rng", format!("{}", table.rng_state())).with("policy_version", table.episodes)
            .with("updates_before", table.updates).with("config", table.config.to_json()).with("dense", table.dense)
            .with("prior", table.prior.clone()))?;
        self.fault(n);
        let t0 = Instant::now();
        let r = episode::run_episode(w, &self.cfg, &self.targets, seed, &mut Policy::Q { table, explore: true })?;
        if r.digest != self.reference_digest {
            return self.failed(n, "training", &r.digest);
        }
        self.journal.append(&self.id, "episode", r.to_json(true).with("attempt", n).with("round", round).with("phase", "train")
            .with("spent_s", t0.elapsed().as_secs_f64()).with("rng_after", format!("{}", table.rng_state()))
            .with("policy_version", table.episodes).with("updates_after", table.updates)
            .with("learned_sha256", learned_digest(table)))?;
        Ok(())
    }

    /// A correctness failure is not an abort: it is journaled and stops the campaign for good.
    fn failed(&mut self, n: u64, phase: &str, digest: &str) -> Result<(), String> {
        self.journal.append(&self.id, "attempt_failed", Value::obj().with("attempt", n).with("phase", phase)
            .with("digest", digest).with("reason", "the job's output differs from the reference"))?;
        Err(format!("{phase} attempt {n}: the job's output differs from the reference; the campaign is stopped"))
    }

    fn holdout(&mut self, w: &mut Workers, round: usize, frozen: &QTable) -> Result<(), String> {
        for j in 1..=self.plan.holdout {
            let seed = self.holdout_seed_of(round, j);
            let burned = self.mine("attempt_aborted").any(|b| b.str("phase") == Some("holdout") && b.num("seed") == Some(seed as f64));
            if burned {
                continue;
            }
            for arm in if j % 2 == 0 { ARMS_EVEN } else { ARMS_ODD } {
                let done = self.mine("episode").any(|b| b.str("phase") == Some("holdout") && b.num("seed") == Some(seed as f64)
                    && b.str("arm") == Some(arm));
                if done {
                    continue;
                }
                let n = self.attempts();
                self.journal.append(&self.id, "attempt", Value::obj().with("attempt", n).with("round", round)
                    .with("phase", "holdout").with("seed", seed).with("arm", arm).with("policy_version", frozen.episodes))?;
                self.fault(n);
                let mut table = frozen.clone();
                let mut policy = if arm.starts_with('F') { Policy::Fixed(self.plan.baseline) } else { Policy::Q { table: &mut table, explore: false } };
                let t0 = Instant::now();
                let r = episode::run_episode(w, &self.cfg, &self.targets, seed, &mut policy)?;
                if r.digest != self.reference_digest {
                    return self.failed(n, "holdout", &r.digest);
                }
                self.journal.append(&self.id, "episode", r.to_json(true).with("attempt", n).with("round", round)
                    .with("phase", "holdout").with("arm", arm).with("spent_s", t0.elapsed().as_secs_f64()))?;
            }
            (self.log)(&format!("round {round}: holdout schedule {j}/{} done", self.plan.holdout));
        }
        Ok(())
    }

    /// The fixed rule (backlog OPT2 as amended, applied per round): paired on complete schedules.
    pub fn decide(&self, round: usize) -> Value {
        let mut by_seed: std::collections::BTreeMap<u64, std::collections::BTreeMap<String, Value>> = Default::default();
        for b in self.mine("episode").filter(|b| b.str("phase") == Some("holdout") && Self::round_of(b) == round) {
            by_seed.entry(b.num("seed").unwrap_or(0.0) as u64).or_default().insert(b.str("arm").unwrap_or("").into(), b.clone());
        }
        decide_rows(&by_seed.values().filter(|m| m.len() == 4).map(|m| {
            let g = |a: &str, k: &str| m[a].num(k).unwrap_or(f64::NAN);
            [g("F", "return"), g("F'", "return"), g("L", "return"), g("L'", "return"),
             g("F", "violations") + g("F'", "violations"), g("L", "violations") + g("L'", "violations"),
             0.5 * (g("F", "wall_s") + g("F'", "wall_s")), 0.5 * (g("L", "wall_s") + g("L'", "wall_s"))]
        }).collect::<Vec<_>>(), self.plan.holdout)
    }

    fn select(&mut self, round: usize) -> Result<(), String> {
        let d0 = Instant::now();
        let trees = self.train_trees();
        let dev = ReplayModel::build(trees.iter().filter(|(s, _)| s % 2 == 0).map(|(_, t)| &t[..]));
        let val = ReplayModel::build(trees.iter().filter(|(s, _)| s % 2 == 1).map(|(_, t)| &t[..]));
        let seeds: Vec<u64> = (1..=3).map(|k| attempt_stream(self.plan.seed, 1_000_000 + 10 * round as u64 + k)).collect();
        let (chosen, report) = dream::select(&dev, &val, self.cfg.threads.len(), 60, 20, &seeds);
        let dream_s = d0.elapsed().as_secs_f64();
        (self.log)(&format!("round {round}: replay selection {} in {dream_s:.2} s",
                            chosen.map_or("kept the current settings (no supported candidate)".into(), |c| c.to_json().dump())));
        self.journal.append(&self.id, "select", report.with("round", round).with("dream_s", dream_s)
            .with("dev_transitions", dev.transitions).with("val_transitions", val.transitions))?;
        Ok(())
    }

    fn activated(&self) -> (bool, bool) {
        (self.mine("activate_intent").next().is_some(), self.mine("activate").next().is_some())
    }

    /// Activate once per campaign: intent, envelope (unless it already holds this policy), done.
    fn activate(&mut self, dir: &Path, round: usize, policy: Value) -> Result<(), String> {
        let (intent, done) = self.activated();
        if done {
            return Ok(());
        }
        if !intent {
            self.journal.append(&self.id, "activate_intent", Value::obj().with("round", round).with("policy", policy.clone()))?;
        }
        let policy = self.mine("activate_intent").next().and_then(|b| b.get("policy")).cloned().unwrap_or(policy);
        let path = dir.join("concurrency-active.json");
        let current = store::read_envelope(&path, CONCURRENCY_SCHEMA, &self.id).ok();
        if current.as_ref().and_then(|c| c.get("policy")) != Some(&policy) {
            activate_concurrency(dir, &self.id, policy)?;
        }
        if Self::has_fault("campaign_crash_activate") {
            std::process::exit(9);
        }
        self.journal.append(&self.id, "activate", Value::obj().with("round", round))?;
        Ok(())
    }

    /// Run (or resume) the campaign to its end; returns the status.
    pub fn run(&mut self, w: &mut Workers, dir: &Path) -> Result<Value, String> {
        if let Some(b) = self.mine("attempt_failed").next() {
            return Err(format!("stopped: attempt {} failed its correctness check; a fresh campaign needs a new --state",
                               b.num("attempt").unwrap_or(-1.0)));
        }
        self.recover()?;
        for round in 1..=self.plan.rounds {
            let decided = self.mine("round_decision").any(|b| Self::round_of(b) == round);
            if !decided {
                while round == 1 && self.plan.warm && !self.sweep_done() {
                    self.sweep_attempt(w)?;
                }
                let config = self.config_for(round);
                let mut table = self.rebuild(config);
                self.verify(&table, round)?;
                let msg = format!("round {round}: learner from {} recorded episodes ({} updates), {:.1} of {} s spent",
                                  table.episodes, table.updates, self.spent(round), self.plan.budget_s);
                (self.log)(&msg);
                while self.spent(round) < self.plan.budget_s {
                    self.train_attempt(w, round, &mut table)?;
                }
                self.holdout(w, round, &table)?;
                let d = self.decide(round);
                #[cfg(feature = "test-faults")]
                let d = if Self::has_fault("campaign_accept") { d.with("verdict", "accepted").with("forced_by_test_fault", true) } else { d };
                let d = d.with("round", round).with("learned_sha256", learned_digest(&table))
                    .with("policy", Frozen::Q(table.clone()).describe(&self.cfg));
                (self.log)(&format!("round {round}: {}", d.str("verdict").unwrap_or("?")));
                self.journal.append(&self.id, "round_decision", d)?;
            }
            let d = self.mine("round_decision").find(|b| Self::round_of(b) == round).cloned().unwrap();
            let (intent, done) = self.activated();
            if (d.str("verdict") == Some("accepted") && !done) || (intent && !done) {
                self.activate(dir, round, d.get("policy").cloned().unwrap_or(Value::Null).with("strategy", "campaign"))?;
            }
            if self.plan.dream && round < self.plan.rounds && !self.mine("select").any(|b| Self::round_of(b) == round) {
                self.select(round)?;
            }
        }
        if self.mine("campaign_done").next().is_none() {
            self.journal.append(&self.id, "campaign_done", self.status())?;
        }
        Ok(self.status())
    }

    pub fn status(&self) -> Value {
        let rounds: Vec<Value> = (1..=self.plan.rounds).map(|r| {
            let eps = |phase: &str| self.mine("episode").filter(|b| b.str("phase") == Some(phase) && Self::round_of(b) == r).count();
            let holdout_s: f64 = self.mine("episode").filter(|b| b.str("phase") == Some("holdout") && Self::round_of(b) == r)
                .filter_map(|b| b.num("spent_s")).sum();
            let burned: Vec<f64> = self.mine("attempt_aborted").filter(|b| b.str("phase") == Some("holdout") && Self::round_of(b) == r)
                .filter_map(|b| b.num("seed")).collect();
            Value::obj().with("round", r).with("train_episodes", eps("train")).with("train_s_charged", self.spent(r))
                .with("holdout_episodes", eps("holdout")).with("holdout_s", holdout_s).with("burned_holdout_schedules", burned)
                .with("config", self.config_for(r).to_json())
                .with("decision", self.mine("round_decision").find(|b| Self::round_of(b) == r).map(|b| {
                    let mut b = b.clone();
                    if let Value::Obj(m) = &mut b { m.remove("policy"); }
                    b
                }))
        }).collect();
        let table = self.rebuild(self.config_for(self.plan.rounds + 1));
        Value::obj().with("plan", self.plan.to_json()).with("attempts", self.attempts())
            .with("aborted", self.mine("attempt_aborted").count()).with("rounds", rounds)
            .with("learned_episodes", table.episodes).with("learned_updates", table.updates)
            .with("activated", self.activated().1).with("done", self.mine("campaign_done").next().is_some())
            .with("failed", self.mine("attempt_failed").count())
    }
}

/// OPT2 phase D: two frozen learners on the same schedules in `F L L' F'` / `L F F' L'` order, `F` the base
/// (the variant without dream), `L` the candidate (with dream); judged by `decide_rows`, so the candidate must
/// beat the base as a learned policy must beat a fixed one. Resumable from its own journal: a completed arm is
/// never rerun, and an interrupted arm burns its schedule.
pub struct Duel<'a> {
    pub journal: &'a mut Journal,
    pub id: String,
    pub cfg: EpisodeConfig,
    pub targets: [f64; 3],
    pub reference_digest: String,
    pub base: QTable,
    pub candidate: QTable,
    pub seeds: Vec<u64>,
    pub log: &'a mut dyn FnMut(&str),
}

impl Duel<'_> {
    fn mine<'b>(&'b self, kind: &'b str) -> impl Iterator<Item = &'b Value> + 'b {
        let id = self.id.clone();
        self.journal.of_kind(kind).filter(move |r| r.str("identity") == Some(id.as_str())).filter_map(|r| r.get("body"))
    }

    fn plan(&self) -> Value {
        Value::obj().with("seeds", self.seeds.iter().map(|s| format!("{s}")).collect::<Vec<_>>())
            .with("base_sha256", learned_digest(&self.base)).with("candidate_sha256", learned_digest(&self.candidate))
            .with("targets_ms", self.targets.to_vec()).with("digest", self.reference_digest.as_str())
    }

    pub fn run(&mut self, w: &mut Workers) -> Result<Value, String> {
        let stored = self.mine("duel_plan").next().cloned();
        match stored {
            Some(p) if p.dump() != self.plan().dump() => return Err("this state holds another duel; a fresh duel needs a new --state".into()),
            Some(_) => {}
            None if !self.journal.records.is_empty() => return Err("this state belongs to another run".into()),
            None => { self.journal.append(&self.id, "duel_plan", self.plan())?; }
        }
        if let Some(d) = self.mine("duel_decision").next() {
            return Ok(d.clone());
        }
        if self.mine("attempt_failed").next().is_some() {
            return Err("stopped: a duel attempt failed its correctness check; a fresh duel needs a new --state".into());
        }
        let closed: std::collections::BTreeSet<u64> = self.mine("episode").chain(self.mine("attempt_aborted"))
            .filter_map(|b| b.num("attempt")).map(|a| a as u64).collect();
        let open: Vec<Value> = self.mine("attempt").filter(|b| b.num("attempt").is_some_and(|a| !closed.contains(&(a as u64))))
            .cloned().collect();
        for b in open {
            self.journal.append(&self.id, "attempt_aborted", b.with("note", "interrupted: its schedule is burned"))?;
        }
        for (j, &seed) in self.seeds.clone().iter().enumerate() {
            if self.mine("attempt_aborted").any(|b| b.num("seed") == Some(seed as f64)) {
                continue;
            }
            for arm in if (j + 1) % 2 == 0 { ARMS_EVEN } else { ARMS_ODD } {
                if self.mine("episode").any(|b| b.num("seed") == Some(seed as f64) && b.str("arm") == Some(arm)) {
                    continue;
                }
                let n = self.mine("attempt").count() as u64;
                self.journal.append(&self.id, "attempt", Value::obj().with("attempt", n).with("seed", seed).with("arm", arm))?;
                let mut table = if arm.starts_with('F') { self.base.clone() } else { self.candidate.clone() };
                let r = episode::run_episode(w, &self.cfg, &self.targets, seed, &mut Policy::Q { table: &mut table, explore: false })?;
                if r.digest != self.reference_digest {
                    self.journal.append(&self.id, "attempt_failed", Value::obj().with("attempt", n).with("digest", r.digest.as_str()))?;
                    return Err(format!("duel attempt {n}: the job's output differs from the reference"));
                }
                self.journal.append(&self.id, "episode", r.to_json(false).with("attempt", n).with("arm", arm))?;
            }
            (self.log)(&format!("duel: schedule {}/{} done", j + 1, self.seeds.len()));
        }
        let mut by_seed: std::collections::BTreeMap<u64, std::collections::BTreeMap<String, Value>> = Default::default();
        for b in self.mine("episode") {
            by_seed.entry(b.num("seed").unwrap_or(0.0) as u64).or_default().insert(b.str("arm").unwrap_or("").into(), b.clone());
        }
        let rows: Vec<[f64; 8]> = by_seed.values().filter(|m| m.len() == 4).map(|m| {
            let g = |a: &str, k: &str| m[a].num(k).unwrap_or(f64::NAN);
            [g("F", "return"), g("F'", "return"), g("L", "return"), g("L'", "return"),
             g("F", "violations") + g("F'", "violations"), g("L", "violations") + g("L'", "violations"),
             0.5 * (g("F", "wall_s") + g("F'", "wall_s")), 0.5 * (g("L", "wall_s") + g("L'", "wall_s"))]
        }).collect();
        let d = decide_rows(&rows, self.seeds.len()).with("base", "variant without dream (F)").with("candidate", "with dream (L)");
        self.journal.append(&self.id, "duel_decision", d.clone())?;
        Ok(d)
    }
}

/// Rows per complete schedule: [F, F', L, L' returns, F+F' violations, L+L' violations, F wall, L wall].
pub fn decide_rows(rows: &[[f64; 8]], planned: usize) -> Value {
    let n = rows.len();
    let f: Vec<f64> = rows.iter().map(|r| 0.5 * (r[0] + r[1])).collect();
    let d: Vec<f64> = rows.iter().map(|r| 0.5 * (r[2] + r[3]) - 0.5 * (r[0] + r[1])).collect();
    let aa: Vec<f64> = rows.iter().flat_map(|r| [(r[1] - r[0]).abs(), (r[3] - r[2]).abs()]).collect();
    let aa_mean = aa.iter().sum::<f64>() / aa.len().max(1) as f64;
    let (pos, neg) = (rows.iter().filter(|r| r[1] > r[0]).count() as u32, rows.iter().filter(|r| r[1] < r[0]).count() as u32);
    let drift_p = (2.0 * sign_test_p(pos.max(neg), pos + neg)).min(1.0);
    let gain = median(&d);
    let wins = d.iter().filter(|x| **x > 0.0).count();
    let need = 0.05 * median(&f).abs();
    let (vf, vl) = (rows.iter().map(|r| r[4]).sum::<f64>(), rows.iter().map(|r| r[5]).sum::<f64>());
    let verdict = if n == 0 || n < (3 * planned).div_ceil(4) {
        "inconclusive (holdout schedules lost to aborts)"
    } else if drift_p < 0.05 {
        "invalid (drift between the A/A runs)"
    } else if gain.abs() <= aa_mean {
        "tie (inside the A/A spread)"
    } else if gain >= need && wins >= (3 * n).div_ceil(4) && vl <= vf {
        "accepted"
    } else {
        "rejected"
    };
    Value::obj().with("schedules", n).with("planned", planned).with("median_paired_gain", gain).with("required_gain", need)
        .with("wins", wins).with("aa_mean_abs", aa_mean).with("drift_sign_p", drift_p).with("violations_fixed", vf)
        .with("violations_learned", vl).with("median_return_fixed", median(&f))
        .with("median_return_learned", median(&rows.iter().map(|r| 0.5 * (r[2] + r[3])).collect::<Vec<_>>()))
        .with("median_job_s_fixed", median(&rows.iter().map(|r| r[6]).collect::<Vec<_>>()))
        .with("median_job_s_learned", median(&rows.iter().map(|r| r[7]).collect::<Vec<_>>()))
        .with("verdict", verdict)
}

/// Write the active concurrency policy (the product's `managed` mode reads it), keeping the previous one.
pub fn activate_concurrency(dir: &Path, id: &str, policy: Value) -> Result<(), String> {
    let path = dir.join("concurrency-active.json");
    let previous = match store::read_envelope(&path, CONCURRENCY_SCHEMA, id) {
        Ok(p) => p.get("policy").cloned().unwrap_or(Value::Null),
        _ => Value::Null,
    };
    store::write_envelope(&path, CONCURRENCY_SCHEMA, id, Value::obj().with("policy", policy).with("previous", previous))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(f: f64, f2: f64, l: f64, l2: f64) -> [f64; 8] {
        [f, f2, l, l2, 0.0, 0.0, 1.0, 1.0]
    }

    #[test]
    fn the_round_rule_needs_complete_schedules_no_drift_and_more_than_the_a_a_spread() {
        let good: Vec<[f64; 8]> = (0..8).map(|i| row(-20.0, -20.0 - (i % 2) as f64, -15.0, -15.0)).collect();
        assert_eq!(decide_rows(&good, 8).str("verdict"), Some("accepted"));
        assert_eq!(decide_rows(&good[..5], 8).str("verdict"), Some("inconclusive (holdout schedules lost to aborts)"));
        // the second fixed run always better: time or order drift, the run is invalid
        let drift: Vec<[f64; 8]> = (0..10).map(|_| row(-20.0, -16.0, -15.0, -15.0)).collect();
        assert_eq!(decide_rows(&drift, 10).str("verdict"), Some("invalid (drift between the A/A runs)"));
        // a gain smaller than what the same policy differs from itself
        let noisy: Vec<[f64; 8]> = (0..8).map(|i| if i % 2 == 0 { row(-20.0, -26.0, -21.0, -15.0) } else { row(-26.0, -20.0, -15.0, -21.0) }).collect();
        assert_eq!(decide_rows(&noisy, 8).str("verdict"), Some("tie (inside the A/A spread)"));
        let mut unsafe_rows = good.clone();
        unsafe_rows[0][5] = 3.0;
        assert_eq!(decide_rows(&unsafe_rows, 8).str("verdict"), Some("rejected"));
        assert_ne!(attempt_stream(1, 2), attempt_stream(1, 3));
        assert_ne!(holdout_seed(1, 1), holdout_seed(2, 1));
    }
}
