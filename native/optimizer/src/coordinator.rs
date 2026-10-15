//! The cycle: detect -> reference -> choose -> run isolated -> journal -> learn -> qualify ->
//! activate or keep the reference -> roll back.
//!
//! The journal is the source of truth: the planner's state is rebuilt from its `test` records every
//! time, and the checkpoint (`planner.json`) is a verified cache, so a crash between a measurement
//! and its checkpoint, or a damaged checkpoint, loses nothing. Records carry the cycle identity
//! (host identity, adapter identity, goal digest). Records of another identity may order the search
//! (transfer) but never count as reference, test outcome or qualification here. An active setting
//! of another identity is rolled back before anything else. The goal (rule, budget, planner
//! settings) is a frozen file whose digest is part of the identity; nothing the learner computes can
//! change it.

use crate::adapter::{self, Adapter, Description};
use crate::identity::{self, Capability, Identity};
use crate::json::{self, Value};
use crate::qualify::{self, Rule, Triplet, Verdict};
use crate::sha256;
use crate::store::{self, EnvelopeError, Journal};
use ironmule_experiment_planner::{Planner, PlannerConfig};
use std::path::{Path, PathBuf};

pub const GOAL_SCHEMA: &str = "imopt.goal.v1";
const PLANNER_SCHEMA: &str = "imopt.planner.v1";
const ACTIVE_SCHEMA: &str = "imopt.active.v1";

#[derive(Clone, Debug)]
pub struct Goal {
    pub rule: Rule,
    /// Qualification protocol: 1 (default; the online controller's) or 2 (OPT4, explicit opt-in).
    pub protocol: u32,
    pub max_tests: usize,
    pub max_seconds: f64,
    pub reference_runs: usize,
    pub planner: PlannerConfig,
    pub digest: String,
    pub raw: Value,
}

impl Goal {
    pub fn load(path: &Path) -> Result<Goal, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("goal unreadable: {e}"))?;
        let v = json::parse(&text).map_err(|e| format!("goal: {e}"))?;
        if v.str("schema") != Some(GOAL_SCHEMA) {
            return Err(format!("goal schema must be {GOAL_SCHEMA}"));
        }
        let r = v.get("rule").ok_or("goal has no rule")?;
        let b = v.get("budget").ok_or("goal has no budget")?;
        let p = v.get("planner").ok_or("goal has no planner settings")?;
        let need = |o: &Value, k: &str| o.num(k).ok_or(format!("goal is missing {k}"));
        let rule = Rule {
            min_gain: need(r, "min_gain")?,
            max_regression: need(r, "max_regression")?,
            pairs: need(r, "pairs")? as u32,
            family_alpha: need(r, "family_alpha")?,
            max_trials: need(r, "max_trials")? as u32,
        };
        if !rule.valid() {
            return Err("goal rule is invalid".into());
        }
        let protocol = match v.get("protocol") {
            None => 1,
            Some(p) => match p.as_f64() {
                Some(x) if x == 1.0 || x == 2.0 => x as u32,
                _ => return Err(format!("goal protocol must be 1 or 2, got {}", p.dump())),
            },
        };
        let planner = PlannerConfig {
            prior_sd: need(p, "prior_sd")?,
            noise_sd: need(p, "noise_sd")?,
            min_gain_per_s: need(p, "min_gain_per_s")?,
            default_cost_s: need(p, "default_cost_s")?,
        };
        // default_cost_s 0 means: an untested action costs what the measured reference costs.
        Planner::new(PlannerConfig { default_cost_s: planner.default_cost_s.max(1.0), ..planner }).ok_or("goal planner settings are invalid")?;
        let goal = Goal {
            rule,
            protocol,
            max_tests: need(b, "max_tests")? as usize,
            max_seconds: need(b, "max_seconds")?,
            reference_runs: (need(b, "reference_runs")? as usize).max(1),
            planner,
            digest: sha256::hex_digest(v.dump().as_bytes()),
            raw: v,
        };
        if goal.max_tests == 0 || !(goal.max_seconds > 0.0) {
            return Err("goal budget must be positive".into());
        }
        Ok(goal)
    }
}

pub struct Coordinator {
    pub dir: PathBuf,
    pub adapter: Adapter,
    pub goal: Goal,
    pub host: Identity,
    pub capabilities: Vec<Capability>,
    pub description: Description,
    pub identity: String,
    pub journal: Journal,
    pub log: Box<dyn FnMut(&str)>,
    /// Median reference cost once measured (the planner's cost of an untested action when the goal says 0).
    pub reference_cost: Option<f64>,
}

#[derive(Debug, PartialEq)]
pub enum Outcome {
    Activated { action: String, mean_ratio: f64 },
    ReferenceKept { reason: String },
}

fn cycle_identity(host: &Identity, d: &Description, goal: &Goal) -> String {
    let v = Value::obj().with("host", host.digest.as_str()).with("adapter", d.name.as_str())
        .with("adapter_identity", d.identity.clone()).with("goal", goal.digest.as_str());
    sha256::hex_digest(v.dump().as_bytes())
}

impl Coordinator {
    pub fn open(dir: &Path, adapter: Adapter, goal_path: &Path, code_digest: Option<&str>) -> Result<Coordinator, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("state dir: {e}"))?;
        let goal = Goal::load(goal_path)?;
        let (host, capabilities) = identity::detect(code_digest);
        let description = adapter.describe().map_err(|e| e.to_string())?;
        let identity = cycle_identity(&host, &description, &goal);
        let journal = Journal::open(&dir.join("journal.jsonl"))?;
        let mut c = Coordinator { dir: dir.to_path_buf(), adapter, goal, host, capabilities, description, identity, journal,
                                  log: Box::new(|m| eprintln!("imopt: {m}")), reference_cost: None };
        if c.journal.torn_recovered {
            c.record("recovered", Value::obj().with("what", "torn final journal line kept aside"))?;
        }
        c.close_interrupted()?;
        Ok(c)
    }

    /// Journal a measurement before it starts; its result record carries the same `attempt` number.
    fn begin(&mut self, what: &str, action: &str) -> Result<u64, String> {
        let n = self.mine("attempt").count() as u64;
        self.record("attempt", Value::obj().with("attempt", n).with("what", what).with("action", action))?;
        Ok(n)
    }

    /// An attempt without its result was interrupted (the run that made it died; by now its processes
    /// are gone, the lease gate saw to that). It is recorded once, never counted as a measurement, and
    /// charged to the budget: its elapsed time until now, at most the adapter timeout (a conservative
    /// bound: the true time is unknown, and a budget must never shrink by a crash).
    fn close_interrupted(&mut self) -> Result<(), String> {
        let closed: std::collections::HashSet<u64> = self.journal.records.iter()
            .filter(|r| r.str("identity") == Some(self.identity.as_str()) && r.str("kind") != Some("attempt"))
            .filter_map(|r| r.get("body").and_then(|b| b.num("attempt"))).map(|n| n as u64).collect();
        let open: Vec<Value> = self.mine("attempt").filter(|r| r.get("body").and_then(|b| b.num("attempt"))
            .is_some_and(|n| !closed.contains(&(n as u64)))).cloned().collect();
        for r in open {
            let b = r.get("body").cloned().unwrap_or(Value::Null);
            let elapsed = (store::now_ms().saturating_sub(r.num("time_ms").unwrap_or(0.0) as u64)) as f64 / 1000.0;
            let charged = elapsed.min(self.adapter.timeout.as_secs_f64());
            self.record("attempt_interrupted", b.with("seconds", charged)
                .with("note", "interrupted: not a measurement; charged its elapsed time, at most the adapter timeout"))?;
            (self.log)(&format!("an interrupted attempt is recorded and charged {charged:.1} s"));
        }
        Ok(())
    }

    fn record(&mut self, kind: &str, body: Value) -> Result<Value, String> {
        self.journal.append(&self.identity, kind, body)
    }

    fn mine<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
        let id = self.identity.clone();
        self.journal.of_kind(kind).filter(move |r| r.str("identity") == Some(id.as_str()))
    }

    fn arm_of(&self, action: &str) -> Option<usize> {
        self.description.actions.iter().position(|a| a.id == action)
    }

    /// Rebuild the planner from the journal (this identity's tests, plus other identities' tests of
    /// the same adapter as ordering-only transfer). The checkpoint is compared, never trusted over it.
    fn planner(&mut self) -> Result<(Planner, usize), String> {
        let mut config = self.goal.planner;
        if config.default_cost_s <= 0.0 {
            config.default_cost_s = self.reference_cost.unwrap_or(1.0);
        }
        let mut planner = Planner::new(config).ok_or("planner settings")?;
        let mut transfer = 0;
        let name = self.description.name.clone();
        for r in self.journal.of_kind("test") {
            let b = r.get("body").unwrap();
            if b.str("adapter") != Some(name.as_str()) {
                continue;
            }
            let Some(arm) = b.str("action").and_then(|a| self.arm_of(a)) else { continue };
            let seconds = b.num("seconds").unwrap_or(0.0);
            let foreign = r.str("identity") != Some(self.identity.as_str());
            match (b.boolean("usable"), b.num("log_effect")) {
                (Some(true), Some(effect)) => { planner.observe(arm, effect, seconds); }
                _ => { planner.fail(arm, seconds); }
            }
            transfer += foreign as usize;
        }
        let path = self.dir.join("planner.json");
        let tests = self.journal.of_kind("test").count();
        let state = Value::obj().with("bytes_sha256", sha256::hex_digest(&planner.to_bytes())).with("tests", tests);
        match store::read_envelope(&path, PLANNER_SCHEMA, &self.identity) {
            Ok(old) if old == state => {}
            // Written at another journal position (normal: tests ran since): nothing to compare.
            Ok(old) if old.num("tests") != Some(tests as f64) => {}
            Ok(_) => { self.record("checkpoint_stale", Value::obj().with("action", "rebuilt from journal"))?; }
            Err(EnvelopeError::Missing) => {}
            Err(e) => {
                let moved = store::quarantine(&path);
                self.record("checkpoint_quarantined", Value::obj().with("error", format!("{e:?}"))
                    .with("moved_to", moved.map(|p| p.display().to_string())))?;
            }
        }
        store::write_envelope(&path, PLANNER_SCHEMA, &self.identity, state)?;
        Ok((planner, transfer))
    }

    /// Before anything else: an active setting must belong to this identity, else it is reverted.
    pub fn enforce_identity(&mut self) -> Result<(), String> {
        let path = self.dir.join("active.json");
        let (previous, foreign_identity) = match std::fs::read_to_string(&path).ok().and_then(|t| json::parse(t.trim()).ok()) {
            None => return Ok(()),
            Some(v) => {
                let previous = v.get("payload").and_then(|p| p.get("previous")).cloned();
                (previous, v.str("identity").map(String::from))
            }
        };
        match store::read_envelope(&path, ACTIVE_SCHEMA, &self.identity) {
            Ok(_) | Err(EnvelopeError::Missing) => Ok(()),
            Err(e) => {
                (self.log)(&format!("active setting not valid for this identity ({e:?}): reverting"));
                let previous = previous.or_else(|| self.last_activation_previous()).unwrap_or(Value::Null);
                self.revert(&previous)?;
                store::quarantine(&path);
                self.record("identity_rollback", Value::obj().with("error", format!("{e:?}"))
                    .with("foreign_identity", foreign_identity))?;
                Ok(())
            }
        }
    }

    fn last_seq(&self, kinds: &[&str]) -> f64 {
        kinds.iter().filter_map(|k| self.mine(k).last().and_then(|r| r.num("seq"))).fold(-1.0, f64::max)
    }

    /// This identity's activation that no rollback, invalidation or outcome has closed yet... or, with
    /// `after_outcome` false, the last activation not yet rolled back (the one `rollback` must undo).
    fn open_activation(&self, after_outcome: bool) -> Option<Value> {
        let mut closed = vec!["rollback", "identity_rollback", "outcome_invalidated"];
        if after_outcome {
            closed.push("outcome");
        }
        let cut = self.last_seq(&closed);
        self.mine("activate").last().filter(|r| r.num("seq").unwrap_or(-1.0) > cut).and_then(|r| r.get("body")).cloned()
    }

    fn last_activation_previous(&self) -> Option<Value> {
        self.open_activation(false).and_then(|b| b.get("previous").cloned())
    }

    fn revert(&mut self, previous: &Value) -> Result<(), String> {
        let answer = self.adapter.call(&["revert", "--previous", &previous.dump()]).map_err(|e| e.to_string())?;
        if answer.boolean("ok") != Some(true) {
            return Err(format!("revert failed: {}", answer.dump()));
        }
        let active = self.adapter.call(&["active"]).map_err(|e| e.to_string())?;
        let now = active.get("action").cloned().unwrap_or(Value::Null);
        let expected_reference = now == Value::Null || now.as_str() == Some(self.description.reference.as_str());
        let restored_previous = previous.get("action").is_some_and(|a| *a == now);
        if !expected_reference && !restored_previous {
            return Err(format!("revert did not take effect: adapter reports {}", now.dump()));
        }
        Ok(())
    }

    fn reference(&mut self) -> Result<(f64, String), String> {
        let reference = self.description.reference.clone();
        let mut done: Vec<(f64, String)> = self.mine("reference").filter_map(|r| {
            let b = r.get("body")?;
            Some((b.num("cost")?, b.str("correctness")?.to_string()))
        }).collect();
        while done.len() < self.goal.reference_runs {
            let n = self.begin("reference", &reference)?;
            let r = adapter::run(&self.adapter, &reference, done.len() as u64);
            if !r.ok {
                self.record("reference_failed", r.to_json().with("attempt", n))?;
                return Err(format!("reference run failed: {:?}", r.error));
            }
            self.record("reference", r.to_json().with("action", reference.as_str()).with("attempt", n))?;
            (self.log)(&format!("reference run {}: cost {:.4}", done.len() + 1, r.cost));
            done.push((r.cost, r.correctness));
        }
        if done.iter().any(|(_, d)| *d != done[0].1) {
            return Err("reference output is not deterministic: nothing can be qualified against it".into());
        }
        let mut costs: Vec<f64> = done.iter().map(|x| x.0).collect();
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.reference_cost = Some(costs[costs.len() / 2]);
        Ok((costs[costs.len() / 2], done[0].1.clone()))
    }

    fn spent_seconds(&self) -> f64 {
        ["reference", "test", "triplet_run", "attempt_interrupted"].iter()
            .flat_map(|k| self.mine(k).filter_map(|r| r.get("body").and_then(|b| b.num("seconds"))).collect::<Vec<_>>())
            .sum()
    }

    /// Explore within the budget: the planner picks the next test by expected gain per second.
    fn explore(&mut self, ref_cost: f64, ref_digest: &str) -> Result<(), String> {
        loop {
            let (planner, transfer) = self.planner()?;
            let tests = self.mine("test").count();
            if tests >= self.goal.max_tests || self.spent_seconds() >= self.goal.max_seconds {
                self.record("explore_end", Value::obj().with("reason", "budget").with("tests", tests))?;
                return Ok(());
            }
            // Each action is screened once here (as RSI1 screens each configuration once); repetition
            // belongs to the qualification. The planner orders the untested ones and decides when to stop.
            let tested: std::collections::HashSet<String> = self.mine("test")
                .filter_map(|r| r.get("body").and_then(|b| b.str("action")).map(String::from)).collect();
            let arms: Vec<u32> = self.description.actions.iter().enumerate()
                .filter(|(_, a)| a.id != self.description.reference && !tested.contains(&a.id)).map(|(i, _)| i as u32).collect();
            if arms.is_empty() {
                self.record("explore_end", Value::obj().with("reason", "every action tested").with("tests", tests))?;
                return Ok(());
            }
            let best = (0..self.description.actions.len()).map(|a| planner.status(a))
                .filter(|s| s.observations > 0).map(|s| s.posterior_mean).fold(0.0f64, f64::min);
            let Some((index, score)) = planner.choose(&arms, &vec![0.0; arms.len()], best) else {
                self.record("explore_end", Value::obj().with("reason", "no test worth its time").with("tests", tests))?;
                return Ok(());
            };
            let action = self.description.actions[arms[index] as usize].clone();
            (self.log)(&format!("test {} ({}): score {:.3} %/s, transfer observations {}", tests + 1, action.id, score, transfer));
            let n = self.begin("test", &action.id)?;
            let r = adapter::run(&self.adapter, &action.id, 1000 + tests as u64);
            let same = r.ok && r.correctness == ref_digest;
            let usable = r.ok && same;
            let effect = if r.ok { (r.cost / ref_cost).ln() } else { f64::NAN };
            self.record("test", r.to_json().with("adapter", self.description.name.as_str()).with("action", action.id.as_str())
                .with("usable", usable).with("same_output", same).with("auto", action.auto)
                .with("log_effect", if usable { Value::from(effect) } else { Value::Null }).with("attempt", n))?;
        }
    }

    /// The best measured auto candidate with identical output and a mean effect below zero.
    fn champion(&mut self) -> Result<Option<String>, String> {
        let (planner, _) = self.planner()?;
        let mut best: Option<(String, f64)> = None;
        let local: std::collections::HashSet<String> = self.mine("test")
            .filter(|r| r.get("body").and_then(|b| b.boolean("usable")) == Some(true))
            .filter_map(|r| r.get("body").and_then(|b| b.str("action")).map(String::from)).collect();
        for (i, a) in self.description.actions.iter().enumerate() {
            if !a.auto || a.id == self.description.reference || !local.contains(&a.id) {
                continue;
            }
            let mean = planner.status(i).posterior_mean;
            if mean < 0.0 && best.as_ref().map_or(true, |(_, m)| mean < *m) {
                best = Some((a.id.clone(), mean));
            }
        }
        Ok(best.map(|b| b.0))
    }

    fn qualify(&mut self, candidate: &str, ref_digest: &str) -> Result<Verdict, String> {
        let trials = self.mine("qualify_start").count() as u32;
        let resumed = self.mine("qualify_start").last()
            .and_then(|r| r.get("body")).and_then(|b| b.str("action")).map(String::from);
        let open = resumed.as_deref() == Some(candidate) && self.mine("qualify_end").count() < self.mine("qualify_start").count();
        if !open {
            if trials >= self.goal.rule.max_trials {
                return Ok(Verdict::Rejected { reason: "trial cap reached".into(), mean_ratio: f64::NAN, mean_noise: f64::NAN, upper_bound: 1.0 });
            }
            self.record("qualify_start", Value::obj().with("action", candidate).with("trial", trials + 1))?;
        }
        if self.description.adapter_evaluator {
            let n = self.begin("qualify", candidate)?;
            let answer = self.adapter.call(&["qualify", "--action", candidate]).map_err(|e| e.to_string())?;
            let verdict = if answer.boolean("ok") == Some(true) && answer.boolean("qualified") == Some(true) {
                Verdict::Qualified { mean_ratio: answer.num("mean_ratio").unwrap_or(f64::NAN), mean_noise: f64::NAN, upper_bound: f64::NAN }
            } else {
                Verdict::Rejected { reason: answer.str("reason").unwrap_or("adapter evaluator rejected").to_string(),
                                    mean_ratio: answer.num("mean_ratio").unwrap_or(f64::NAN), mean_noise: f64::NAN, upper_bound: 1.0 }
            };
            let ratio = match &verdict { Verdict::Qualified { mean_ratio, .. } | Verdict::Rejected { mean_ratio, .. } => *mean_ratio, _ => f64::NAN };
            self.record("qualify_end", Value::obj().with("action", candidate).with("verdict", format!("{verdict:?}"))
                .with("mean_ratio", ratio).with("evaluator", "adapter").with("evidence", answer).with("attempt", n))?;
            return Ok(verdict);
        }
        let reference = self.description.reference.clone();
        let start_seq = self.mine("qualify_start").last().and_then(|r| r.num("seq")).unwrap_or(0.0);
        let mut triplets: Vec<Triplet> = self.mine("triplet").filter(|r| r.num("seq").unwrap_or(0.0) > start_seq)
            .filter_map(|r| {
                // A failed run was journaled as null: it stays in as NaN, which the rule rejects.
                let b = r.get("body")?;
                let x = |k: &str| b.num(k).unwrap_or(f64::NAN);
                Some(Triplet { a: x("a"), b: x("b"), aa: x("aa"), correct: b.boolean("correct").unwrap_or(false) })
            }).collect();
        loop {
            match qualify::judge_protocol(self.goal.protocol, &self.goal.rule, &triplets) {
                Verdict::Incomplete => {}
                verdict => {
                    let ratio = match &verdict { Verdict::Qualified { mean_ratio, .. } | Verdict::Rejected { mean_ratio, .. } => *mean_ratio, _ => f64::NAN };
                    self.record("qualify_end", Value::obj().with("action", candidate).with("verdict", format!("{verdict:?}")).with("mean_ratio", ratio)
                        .with("evaluator", if self.goal.protocol == 2 { "coordinator A/A/B protocol 2 (OPT4)" }
                              else { "coordinator A/A/B (online controller protocol 1)" }).with("triplets", triplets.len()))?;
                    return Ok(verdict);
                }
            }
            let k = triplets.len();
            let n = self.begin("triplet", candidate)?;
            let (t, order, runs) = collect_triplet(&self.adapter, &self.adapter, &reference, candidate, k, ref_digest);
            for (slot, action, r) in runs {
                self.record("triplet_run", r.to_json().with("slot", slot).with("action", action).with("attempt", n))?;
            }
            self.record("triplet", Value::obj().with("a", t.a).with("b", t.b).with("aa", t.aa).with("correct", t.correct)
                .with("order", order.join(",")).with("action", candidate).with("attempt", n))?;
            (self.log)(&format!("triplet {}: R = {:.4}", k + 1, t.b / (0.5 * (t.a + t.aa))));
            triplets.push(t);
        }
    }

    fn activate(&mut self, action: &str, ref_digest: &str, mean_ratio: f64) -> Result<Outcome, String> {
        let answer = self.adapter.call(&["apply", "--action", action]).map_err(|e| e.to_string())?;
        if answer.boolean("ok") != Some(true) {
            self.record("activate_failed", answer)?;
            return Ok(Outcome::ReferenceKept { reason: "adapter refused to apply".into() });
        }
        let previous = answer.get("previous").cloned().unwrap_or(Value::Null);
        self.record("activate", Value::obj().with("action", action).with("previous", previous.clone()))?;
        store::write_envelope(&self.dir.join("active.json"), ACTIVE_SCHEMA, &self.identity,
                              Value::obj().with("action", action).with("previous", previous.clone()))?;
        self.recheck(action, ref_digest, mean_ratio)
    }

    /// After activation: the adapter must report the action, and it must still give the reference output.
    fn recheck(&mut self, action: &str, ref_digest: &str, mean_ratio: f64) -> Result<Outcome, String> {
        let active = self.adapter.call(&["active"]).map_err(|e| e.to_string())?;
        let n = self.begin("recheck", action)?;
        let r = adapter::run(&self.adapter, action, 9000);
        let ok = active.str("action") == Some(action) && r.ok && r.correctness == ref_digest;
        self.record("recheck", r.to_json().with("active_reported", active.get("action").cloned().unwrap_or(Value::Null))
            .with("passed", ok).with("attempt", n))?;
        if !ok {
            self.rollback("recheck failed")?;
            return Ok(Outcome::ReferenceKept { reason: "post-activation recheck failed, rolled back".into() });
        }
        Ok(Outcome::Activated { action: action.to_string(), mean_ratio })
    }

    pub fn rollback(&mut self, reason: &str) -> Result<bool, String> {
        let path = self.dir.join("active.json");
        let previous = match store::read_envelope(&path, ACTIVE_SCHEMA, &self.identity) {
            Ok(p) => p.get("previous").cloned().unwrap_or(Value::Null),
            // Envelope lost or damaged: the journal still knows the open activation and its previous state.
            Err(_) => match self.last_activation_previous() {
                Some(p) => p,
                None => {
                    self.record("rollback", Value::obj().with("reason", reason).with("result", "nothing active"))?;
                    return Ok(false);
                }
            },
        };
        self.revert(&previous)?;
        let moved = store::quarantine(&path);
        self.record("rollback", Value::obj().with("reason", reason).with("previous", previous)
            .with("result", "reverted").with("active_file", moved.map(|p| p.display().to_string())))?;
        Ok(true)
    }

    pub fn cycle(&mut self) -> Result<Outcome, String> {
        self.enforce_identity()?;
        let caps: Vec<Value> = self.capabilities.iter().map(Capability::to_json).collect();
        if self.mine("detect").next().is_none() {
            self.record("detect", Value::obj().with("host", self.host.facts.clone()).with("host_digest", self.host.digest.as_str())
                .with("hardware_class", self.host.hardware_class.as_str()).with("capabilities", caps)
                .with("adapter", self.description.raw.clone()).with("goal", self.goal.raw.clone()))?;
        }
        let (ref_cost, ref_digest) = self.reference()?;
        // A crash after `activate` and before `outcome`: finish that activation, never apply again.
        if let Some(open) = self.open_activation(true) {
            let action = open.str("action").unwrap_or("").to_string();
            (self.log)(&format!("resuming the activation of {action} interrupted before its outcome"));
            if store::read_envelope(&self.dir.join("active.json"), ACTIVE_SCHEMA, &self.identity).is_err() {
                store::write_envelope(&self.dir.join("active.json"), ACTIVE_SCHEMA, &self.identity,
                    Value::obj().with("action", action.as_str()).with("previous", open.get("previous").cloned().unwrap_or(Value::Null)))?;
            }
            let mean_ratio = self.mine("qualify_end").last().and_then(|r| r.get("body")).and_then(|b| b.num("mean_ratio"))
                .unwrap_or(f64::NAN);
            let outcome = self.recheck(&action, &ref_digest, mean_ratio)?;
            return self.finish(outcome);
        }
        let last_rollback = self.last_seq(&["rollback", "identity_rollback", "outcome_invalidated"]);
        let (last_outcome, last_explore) = (self.last_seq(&["outcome"]), self.last_seq(&["explore_end"]));
        let mut last_rollback = last_rollback;
        if last_outcome > last_rollback {
            match self.mine("outcome").last().map(outcome_from) {
                Some(Ok(Outcome::Activated { action, mean_ratio })) => {
                    // An activated outcome stands only while the adapter still reports it.
                    let now = self.adapter.call(&["active"]).map_err(|e| e.to_string())?;
                    if now.str("action") == Some(action.as_str()) {
                        (self.log)("a finished cycle exists for this identity; its outcome stands (roll back to rerun)");
                        return Ok(Outcome::Activated { action, mean_ratio });
                    }
                    let rec = self.record("outcome_invalidated", Value::obj().with("action", action.as_str())
                        .with("adapter_reports", now.get("action").cloned().unwrap_or(Value::Null)))?;
                    last_rollback = rec.num("seq").unwrap_or(last_rollback);
                }
                Some(Ok(o)) => {
                    (self.log)("a finished cycle exists for this identity; its outcome stands (roll back to rerun)");
                    return Ok(o);
                }
                _ => {}
            }
        }
        if last_explore <= last_rollback.max(if last_outcome > last_rollback { last_outcome } else { -1.0 }) {
            self.explore(ref_cost, &ref_digest)?;
        }
        let outcome = match self.champion()? {
            None => Outcome::ReferenceKept { reason: "no measured candidate with identical output beat the reference".into() },
            Some(candidate) => match self.qualify(&candidate, &ref_digest)? {
                Verdict::Qualified { mean_ratio, .. } => self.activate(&candidate, &ref_digest, mean_ratio)?,
                Verdict::Rejected { reason, .. } => Outcome::ReferenceKept { reason: format!("{candidate} not qualified: {reason}") },
                Verdict::Incomplete => unreachable!(),
            },
        };
        self.finish(outcome)
    }

    fn finish(&mut self, outcome: Outcome) -> Result<Outcome, String> {
        let body = match &outcome {
            Outcome::Activated { action, mean_ratio } => Value::obj().with("result", "activated").with("action", action.as_str()).with("mean_ratio", *mean_ratio),
            Outcome::ReferenceKept { reason } => Value::obj().with("result", "reference_kept").with("reason", reason.as_str()),
        };
        self.record("outcome", body.with("spent_seconds", self.spent_seconds()))?;
        Ok(outcome)
    }

    /// Each adapter action's state on this identity: controllable (offered), tested (a usable measurement),
    /// qualified (the evaluator passed it), or unsupported (its test failed as unsupported).
    pub fn action_states(&self) -> Vec<Value> {
        self.description.actions.iter().map(|a| {
            let tests: Vec<&Value> = self.mine("test").filter_map(|r| r.get("body")).filter(|b| b.str("action") == Some(a.id.as_str())).collect();
            let qualified = self.mine("qualify_end").filter_map(|r| r.get("body"))
                .any(|b| b.str("action") == Some(a.id.as_str()) && b.str("verdict").is_some_and(|v| v.starts_with("Qualified")));
            let state = if qualified {
                identity::CapState::Qualified
            } else if tests.iter().any(|b| b.boolean("usable") == Some(true)) {
                identity::CapState::Tested
            } else if tests.iter().any(|b| b.str("error").is_some_and(|e| e.contains("unsupported"))) {
                identity::CapState::Unsupported
            } else {
                identity::CapState::Controllable
            };
            Value::obj().with("action", a.id.as_str()).with("state", state.name()).with("auto", a.auto)
        }).collect()
    }

    pub fn status(&self) -> Value {
        let tests: Vec<Value> = self.mine("test").filter_map(|r| r.get("body").cloned()).collect();
        let active = store::read_envelope(&self.dir.join("active.json"), ACTIVE_SCHEMA, &self.identity).ok();
        Value::obj().with("identity", self.identity.as_str()).with("adapter", self.description.name.as_str())
            .with("hardware_class", self.host.hardware_class.as_str())
            .with("capabilities", self.capabilities.iter().map(Capability::to_json).collect::<Vec<_>>())
            .with("action_capabilities", self.action_states())
            .with("journal_records", self.journal.records.len())
            .with("tests", tests.len())
            .with("active", active.unwrap_or(Value::Null))
            .with("outcome", self.mine("outcome").last().and_then(|r| r.get("body")).cloned().unwrap_or(Value::Null))
    }
}

fn outcome_from(r: &Value) -> Result<Outcome, ()> {
    let b = r.get("body").ok_or(())?;
    match b.str("result") {
        Some("activated") => Ok(Outcome::Activated { action: b.str("action").ok_or(())?.to_string(), mean_ratio: b.num("mean_ratio").unwrap_or(f64::NAN) }),
        Some("reference_kept") => Ok(Outcome::ReferenceKept { reason: b.str("reason").unwrap_or("").to_string() }),
        _ => Err(()),
    }
}

/// One A/A/B triplet as the qualification runs it: order `a, b, aa` for even `k`, `b, a, aa` for odd
/// (AA always last), seeds `5000 + 3k + slot`, each run a fresh adapter process; correct only if every
/// run succeeded with the reference's output. The study harness (OPT4) uses the same function, with a
/// separate adapter for the reference arms when it pads them.
pub fn collect_triplet(reference_adapter: &Adapter, candidate_adapter: &Adapter, reference: &str, candidate: &str,
                       k: usize, ref_digest: &str) -> (Triplet, [&'static str; 3], Vec<(&'static str, String, adapter::RunResult)>) {
    let order: [&'static str; 3] = if k % 2 == 0 { ["a", "b", "aa"] } else { ["b", "a", "aa"] };
    let mut got = std::collections::BTreeMap::new();
    let mut correct = true;
    let mut runs = Vec::new();
    for slot in order {
        let (who, action) = if slot == "b" { (candidate_adapter, candidate) } else { (reference_adapter, reference) };
        let r = adapter::run(who, action, 5000 + 3 * k as u64 + got.len() as u64);
        correct &= r.ok && r.correctness == ref_digest;
        got.insert(slot, if r.ok { r.cost } else { f64::NAN });
        runs.push((slot, action.to_string(), r));
    }
    (Triplet { a: got["a"], b: got["b"], aa: got["aa"], correct }, order, runs)
}

#[cfg(test)]
mod goal_tests {
    use super::*;

    fn goal_with(extra: &str) -> Result<Goal, String> {
        let d = std::env::temp_dir().join(format!("imopt-goal-{}-{}", std::process::id(), extra.len()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("goal.json");
        std::fs::write(&p, format!(r#"{{"schema":"imopt.goal.v1",{extra}
            "rule":{{"min_gain":0.03,"max_regression":0.1,"pairs":8,"family_alpha":0.05,"max_trials":4}},
            "budget":{{"max_tests":8,"max_seconds":60,"reference_runs":2}},
            "planner":{{"prior_sd":0.5,"noise_sd":0.05,"min_gain_per_s":0.0,"default_cost_s":0}}}}"#)).unwrap();
        Goal::load(&p)
    }

    /// Protocol 2 only by explicit choice; every existing goal (no field) stays protocol 1.
    #[test]
    fn the_protocol_is_explicit_and_unknown_versions_are_refused() {
        assert_eq!(goal_with("").unwrap().protocol, 1);
        assert_eq!(goal_with(r#""protocol":1,"#).unwrap().protocol, 1);
        assert_eq!(goal_with(r#""protocol":2,"#).unwrap().protocol, 2);
        for bad in [r#""protocol":3,"#, r#""protocol":1.5,"#, r#""protocol":"2","#, r#""protocol":0,"#] {
            assert!(goal_with(bad).unwrap_err().contains("protocol must be 1 or 2"), "{bad}");
        }
        for shipped in ["goals/cpujob.json", "goals/ironmule.json"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(shipped);
            assert_eq!(Goal::load(&path).unwrap().protocol, 1, "{shipped}");
        }
    }
}
