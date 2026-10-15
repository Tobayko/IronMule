//! OPT4b study harness: protocol 1 and protocol 2 judged on the same triplets.
//!
//! Each cycle collects exactly `pairs` complete A/A/B triplets through the qualification's own
//! `collect_triplet` (same order alternation and seeds) and never stops early: a production cycle
//! would stop at protocol 1's first veto, but here every triplet is collected and both protocols judge
//! the whole set; the index where production would have stopped is recorded beside it.
//!
//! Effects (backlog OPT4b): `null` (candidate = reference, 1 thread), `sub` (the reference arms sleep
//! 0.0101 x their measured job time after the job through `IMOPT_CPUJOB_PAD`, and report a cost computed
//! as 1.0101 x that job time; outputs identical: a synthetic 1 % gain below `min_gain`), `large` (the
//! largest thread count of the adapter's set on this host). Conditions: `ambient`, `steady` (a `hog`
//! process with `h` = CPUs of the fastest core class - 1 threads, so the 1-thread arms and the hog
//! fill those cores: the performance cores on Apple silicon, all logical CPUs elsewhere),
//! `burst` (that hog switched on and off for exponential times, mean 1 s, seeded per cycle, independent
//! of the slots), `ramp` (a sawtooth: 0, h/2, h, 3h/2 hog threads, one step every 2.5 s). Each cycle
//! starts its triplets 0.4 s after the load, so the first slot does not meet a starting hog. Load is driven by time, never by the arm being
//! measured: load that hits only the reference arms is the known structural blind spot of every rule
//! on (A, B, AA) and is excluded by design, not tested.
//!
//! Journal: `rulecheck_plan`, then per cycle `cycle_start` and `cycle`. A `cycle_start` without its
//! `cycle` was interrupted: it becomes `cycle_aborted` on restart, is never rerun, and its cell keeps
//! one cycle fewer (the analysis then says so).

use crate::adapter::Adapter;
use crate::coordinator::collect_triplet;
use crate::json::Value;
use crate::qualify::{self, Rule, Triplet, Verdict};
use crate::rl::stream;
use crate::store::Journal;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SUB_PAD: f64 = 0.0101;
pub const EFFECTS: [&str; 3] = ["null", "sub", "large"];
pub const CONDITIONS: [&str; 4] = ["ambient", "steady", "burst", "ramp"];

pub type Cell = (&'static str, &'static str);

/// The cells of a run: the full matrix, or the quiet-machine control (null and large, ambient and steady).
pub fn cells(control: bool) -> Vec<Cell> {
    let effects: &[&'static str] = if control { &["null", "large"] } else { &EFFECTS };
    let conditions: &[&'static str] = if control { &["ambient", "steady"] } else { &CONDITIONS };
    effects.iter().flat_map(|e| conditions.iter().map(move |c| (*e, *c))).collect()
}

/// One round per cycle, each round every cell once in a seeded shuffled order, so slow drift over the
/// run spreads evenly and no cell always follows the same one.
pub fn schedule(cells: &[Cell], cycles: usize, seed: u64) -> Vec<Cell> {
    (0..cycles).flat_map(|r| {
        let mut order = cells.to_vec();
        let mut s = stream(seed ^ stream(1_000_000 + r as u64));
        for i in (1..order.len()).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            order.swap(i, (s % (i as u64 + 1)) as usize);
        }
        order
    }).collect()
}

/// Injected hog threads at `t` seconds into a cycle; `h` is the fastest core class's CPUs - 1.
pub fn hog_threads(condition: &str, h: usize, seed: u64, t: f64) -> usize {
    match condition {
        "steady" => h,
        "burst" => {
            let mut s = stream(seed);
            let (mut edge, mut on) = (0.0, false);
            while edge <= t {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
                edge += -u.ln(); // exponential, mean 1 s
                on = !on;
            }
            if on { h } else { 0 }
        }
        "ramp" => [0, h / 2, h, (3 * h) / 2][((t / 2.5) as usize) % 4],
        _ => 0,
    }
}

/// A background thread that keeps a `hog` process at the planned thread count until stopped.
pub struct Load {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// hog processes started and failed to start (a failed one would leave the condition without load)
    counts: Arc<[AtomicUsize; 2]>,
}

impl Load {
    pub fn start(worker: &str, condition: &'static str, h: usize, seed: u64) -> Load {
        let stop = Arc::new(AtomicBool::new(false));
        let counts: Arc<[AtomicUsize; 2]> = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let (flag, worker, tally) = (stop.clone(), worker.to_string(), counts.clone());
        let handle = std::thread::spawn(move || {
            let t0 = Instant::now();
            let (mut child, mut current): (Option<Child>, usize) = (None, 0);
            while !flag.load(Ordering::SeqCst) {
                let want = hog_threads(condition, h, seed, t0.elapsed().as_secs_f64());
                if want != current {
                    if let Some(mut c) = child.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                    if want > 0 {
                        let _lock = crate::adapter::SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                        child = Command::new(&worker).args(["hog", &want.to_string(), &std::process::id().to_string()]).stdin(Stdio::null())
                            .stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok();
                        tally[if child.is_some() { 0 } else { 1 }].fetch_add(1, Ordering::SeqCst);
                    }
                    current = want;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            if let Some(mut c) = child.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
        });
        Load { stop, handle: Some(handle), counts }
    }

    /// Stop the load; (hog processes started, failed to start).
    pub fn finish(mut self) -> (usize, usize) {
        self.halt();
        (self.counts[0].load(Ordering::SeqCst), self.counts[1].load(Ordering::SeqCst))
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.halt();
    }
}

fn verdict_json(v: &Verdict) -> Value {
    match v {
        Verdict::Qualified { mean_ratio, mean_noise, upper_bound } => Value::obj().with("verdict", "qualified")
            .with("mean_ratio", *mean_ratio).with("mean_noise", *mean_noise).with("upper_bound", *upper_bound),
        Verdict::Rejected { reason, mean_ratio, mean_noise, upper_bound } => Value::obj().with("verdict", "rejected")
            .with("reason", reason.as_str()).with("mean_ratio", *mean_ratio).with("mean_noise", *mean_noise).with("upper_bound", *upper_bound),
        Verdict::Incomplete => Value::obj().with("verdict", "incomplete"),
    }
}

/// Both protocols on the same complete triplets; where production protocol 1 would have stopped.
pub fn judge_cycle(rule: &Rule, ts: &[Triplet]) -> Value {
    let p1 = qualify::judge(rule, ts);
    let p2 = qualify::judge_v2(rule, ts);
    let stop = (1..=ts.len()).find(|&m| matches!(qualify::judge(rule, &ts[..m]), Verdict::Rejected { .. }));
    Value::obj().with("triplets", ts.len()).with("p1", verdict_json(&p1)).with("p2", verdict_json(&p2))
        .with("p1_production_stop_after", stop)
}

fn accepted(judged: &Value, protocol: &str) -> bool {
    judged.get(protocol).and_then(|v| v.str("verdict")) == Some("qualified")
}

/// Per cell: complete cycles, acceptances by each protocol, disagreements, drift calls; then the
/// fixed OPT4b criteria. `judged` holds (effect, condition, judge_cycle output) per complete cycle.
pub fn analyse(judged: &[(String, String, Value)], plan: &[Cell], planned_per_cell: usize, need_large: usize) -> Value {
    let mut rows = Vec::new();
    let (mut false_accepts, mut large_short, mut lost) = (0usize, Vec::new(), 0usize);
    for &(effect, condition) in plan {
        let mine: Vec<&Value> = judged.iter().filter(|(e, c, _)| e == effect && c == condition).map(|(_, _, j)| j).collect();
        let n = mine.len();
        let a1 = mine.iter().filter(|j| accepted(j, "p1")).count();
        let a2 = mine.iter().filter(|j| accepted(j, "p2")).count();
        let only2 = mine.iter().filter(|j| accepted(j, "p2") && !accepted(j, "p1")).count();
        let only1 = mine.iter().filter(|j| accepted(j, "p1") && !accepted(j, "p2")).count();
        let drift = mine.iter().filter(|j| j.get("p2").and_then(|v| v.str("reason")).is_some_and(|r| r.starts_with("invalid: drift"))).count();
        let noisy = mine.iter().filter(|j| j.get("p1").and_then(|v| v.str("reason")).is_some_and(|r| r.starts_with("noisy"))).count();
        let unstable = mine.iter().filter(|j| j.get("p2").and_then(|v| v.str("reason")).is_some_and(|r| r.starts_with("unstable"))).count();
        lost += planned_per_cell.saturating_sub(n);
        if effect != "large" {
            false_accepts += a2;
        } else if condition != "ramp" && a2 + planned_per_cell.saturating_sub(n) < need_large {
            // even the lost cycles accepted could not reach the requirement
            large_short.push(format!("{effect}/{condition}: {a2} of {n}"));
        }
        rows.push(Value::obj().with("effect", effect).with("condition", condition).with("cycles", n)
            .with("p1_accepted", a1).with("p2_accepted", a2).with("p2_only", only2).with("p1_only", only1)
            .with("p2_invalid_drift", drift).with("p1_noisy_veto", noisy).with("p2_unstable_veto", unstable));
    }
    // A failed criterion decides even with lost cycles; only then can lost cycles make it inconclusive.
    let verdict = if false_accepts > 0 || !large_short.is_empty() {
        "protocol 2 rejected".to_string()
    } else if lost > 0 {
        "inconclusive (cycles lost to aborts)".to_string()
    } else {
        "protocol 2 passes".to_string()
    };
    Value::obj().with("cells", rows).with("p2_false_accepts", false_accepts).with("large_short", large_short)
        .with("cycles_lost", lost).with("need_large", need_large).with("verdict", verdict.as_str())
}

pub struct Setup<'a> {
    pub journal: &'a mut Journal,
    pub id: String,
    pub worker: String,
    pub rule: Rule,
    pub large_threads: usize,
    pub hog: usize,
    pub seed: u64,
    pub plan: Vec<Cell>,
    pub reference_digest: String,
    pub log: &'a mut dyn FnMut(&str),
    /// The state's lease: runs are supervised (see `proc`) when set.
    pub lease: Option<std::path::PathBuf>,
}

impl Setup<'_> {
    fn mine<'b>(&'b self, kind: &'b str) -> impl Iterator<Item = &'b Value> + 'b {
        let id = self.id.clone();
        self.journal.of_kind(kind).filter(move |r| r.str("identity") == Some(id.as_str())).filter_map(|r| r.get("body"))
    }

    /// Run every planned cycle not yet started; interrupted ones are recorded and skipped.
    pub fn run(&mut self) -> Result<(), String> {
        let closed: std::collections::BTreeSet<u64> = self.mine("cycle").chain(self.mine("cycle_aborted"))
            .filter_map(|b| b.num("index")).map(|x| x as u64).collect();
        let open: Vec<Value> = self.mine("cycle_start").filter(|b| b.num("index").is_some_and(|i| !closed.contains(&(i as u64))))
            .cloned().collect();
        for b in open {
            self.journal.append(&self.id, "cycle_aborted", b.with("note", "interrupted: not judged, never rerun"))?;
        }
        let started: std::collections::BTreeSet<u64> = self.mine("cycle_start").filter_map(|b| b.num("index")).map(|x| x as u64).collect();
        let mut plain = Adapter::new(vec![self.worker.clone()], Duration::from_secs(600));
        let mut padded = Adapter::new(vec!["env".into(), format!("IMOPT_CPUJOB_PAD={SUB_PAD}"), self.worker.clone()], Duration::from_secs(600));
        if let Some(lease) = &self.lease {
            let exe = std::env::current_exe().map_err(|e| format!("cannot supervise the runs: {e}"))?;
            plain = plain.supervised(exe.clone(), lease.clone())?;
            padded = padded.supervised(exe, lease.clone())?;
        }
        let large = format!("threads={}", self.large_threads);
        for (i, &(effect, condition)) in self.plan.clone().iter().enumerate() {
            if started.contains(&(i as u64)) {
                continue;
            }
            let seed = stream(self.seed ^ stream(i as u64));
            self.journal.append(&self.id, "cycle_start", Value::obj().with("index", i).with("effect", effect)
                .with("condition", condition).with("load_seed", format!("{seed}")))?;
            if std::env::var("IMOPT_FAULT").ok().is_some_and(|f| f.split(',').any(|x| x == format!("rulecheck_abort:{i}"))) {
                std::process::exit(9);
            }
            let t0 = Instant::now();
            let load_average = crate::identity::load_average();
            let load = Load::start(&self.worker, condition, self.hog, seed);
            std::thread::sleep(Duration::from_millis(400));
            let (ref_adapter, candidate) = match effect {
                "sub" => (&padded, "threads=1"),
                "large" => (&plain, large.as_str()),
                _ => (&plain, "threads=1"),
            };
            let mut ts = Vec::new();
            let mut costs = Vec::new();
            for k in 0..self.rule.pairs as usize {
                let (t, order, runs) = collect_triplet(ref_adapter, &plain, "threads=1", candidate, k, &self.reference_digest);
                costs.push(Value::obj().with("a", t.a).with("b", t.b).with("aa", t.aa).with("correct", t.correct)
                    .with("order", order.join(",")).with("runs", runs.iter().map(|(slot, action, r)|
                        r.to_json().with("slot", *slot).with("action", action.as_str())).collect::<Vec<_>>()));
                ts.push(t);
            }
            let (hogs, hog_failures) = load.finish();
            let judged = judge_cycle(&self.rule, &ts);
            (self.log)(&format!("cycle {}/{} {effect}/{condition}: p1 {} p2 {} ({:.1} s)", i + 1, self.plan.len(),
                judged.get("p1").and_then(|v| v.str("verdict")).unwrap_or("?"), judged.get("p2").and_then(|v| v.str("verdict")).unwrap_or("?"),
                t0.elapsed().as_secs_f64()));
            self.journal.append(&self.id, "cycle", Value::obj().with("index", i).with("effect", effect).with("condition", condition)
                .with("triplets", costs).with("judged", judged).with("seconds", t0.elapsed().as_secs_f64())
                .with("load_average_at_start", load_average).with("hog_processes", hogs).with("hog_spawn_failures", hog_failures))?;
        }
        Ok(())
    }

    pub fn analysis(&self, cells: &[Cell], planned_per_cell: usize, need_large: usize) -> Value {
        let judged: Vec<(String, String, Value)> = self.mine("cycle").map(|b| (b.str("effect").unwrap_or("").to_string(),
            b.str("condition").unwrap_or("").to_string(), b.get("judged").cloned().unwrap_or(Value::Null))).collect();
        analyse(&judged, cells, planned_per_cell, need_large)
            .with("aborted", self.mine("cycle_aborted").count())
            .with("cycles_with_hog_spawn_failures", self.mine("cycle").filter(|b| b.num("hog_spawn_failures").is_some_and(|f| f > 0.0)).count())
            .with("note", "sub is a synthetic, output-identical 1 % gain (padded reference arms); every other number is a real measurement")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: Rule = Rule { min_gain: 0.03, max_regression: 0.10, pairs: 8, family_alpha: 0.05, max_trials: 4 };

    fn t(a: f64, b: f64, aa: f64) -> Triplet {
        Triplet { a, b, aa, correct: true }
    }

    #[test]
    fn the_plan_rotates_and_the_load_is_driven_by_time_alone() {
        assert_eq!(cells(false).len(), 12);
        assert_eq!(cells(true), vec![("null", "ambient"), ("null", "steady"), ("large", "ambient"), ("large", "steady")]);
        let s = schedule(&cells(true), 3, 7);
        assert_eq!(s.len(), 12);
        for r in 0..3 {
            let mut round = s[4 * r..4 * r + 4].to_vec();
            round.sort();
            let mut all = cells(true);
            all.sort();
            assert_eq!(round, all, "every cell once per round");
        }
        assert_eq!(s, schedule(&cells(true), 3, 7));
        assert_ne!(schedule(&cells(false), 4, 7), schedule(&cells(false), 4, 8));
        assert_eq!(hog_threads("steady", 5, 1, 7.0), 5);
        assert_eq!(hog_threads("ambient", 5, 1, 7.0), 0);
        assert_eq!((0..4).map(|k| hog_threads("ramp", 4, 1, 2.5 * k as f64 + 0.1)).collect::<Vec<_>>(), vec![0, 2, 4, 6]);
        let burst: Vec<usize> = (0..200).map(|k| hog_threads("burst", 3, 42, 0.1 * k as f64)).collect();
        assert_eq!(burst, (0..200).map(|k| hog_threads("burst", 3, 42, 0.1 * k as f64)).collect::<Vec<_>>());
        assert!(burst.contains(&0) && burst.contains(&3));
        assert_ne!(burst, (0..200).map(|k| hog_threads("burst", 3, 43, 0.1 * k as f64)).collect::<Vec<_>>());
    }

    /// Synthetic: protocol 1 would stop production at triplet 2 (noisy), yet both protocols judge all 8.
    #[test]
    fn an_early_production_veto_does_not_shorten_the_shared_data() {
        let mut ts: Vec<Triplet> = (0..8).map(|_| t(10.0, 2.0, 10.0)).collect();
        ts[1] = t(10.0, 2.0, 13.0);
        let j = judge_cycle(&RULE, &ts);
        assert_eq!(j.num("triplets"), Some(8.0));
        assert_eq!(j.num("p1_production_stop_after"), Some(2.0));
        assert_eq!(j.get("p1").and_then(|v| v.str("verdict")), Some("rejected"));
        assert_eq!(j.get("p2").and_then(|v| v.str("verdict")), Some("qualified"));
    }

    #[test]
    fn the_criteria_need_no_false_accept_enough_large_wins_and_every_cycle() {
        let plan = cells(true);
        let row = |e: &str, c: &str, p1: &str, p2: &str| (e.to_string(), c.to_string(),
            Value::obj().with("p1", Value::obj().with("verdict", p1)).with("p2", Value::obj().with("verdict", p2)));
        let mut ok = Vec::new();
        for _ in 0..3 {
            ok.push(row("null", "ambient", "rejected", "rejected"));
            ok.push(row("null", "steady", "rejected", "rejected"));
            ok.push(row("large", "ambient", "rejected", "qualified"));
            ok.push(row("large", "steady", "qualified", "qualified"));
        }
        assert_eq!(analyse(&ok, &plan, 3, 3).str("verdict"), Some("protocol 2 passes"));
        let mut fa = ok.clone();
        fa[0] = row("null", "ambient", "rejected", "qualified");
        assert_eq!(analyse(&fa, &plan, 3, 3).str("verdict"), Some("protocol 2 rejected"));
        assert_eq!(analyse(&ok[1..], &plan, 3, 3).str("verdict"), Some("inconclusive (cycles lost to aborts)"));
        // a failed criterion decides even when cycles were lost
        assert_eq!(analyse(&fa[..fa.len() - 1], &plan, 3, 3).str("verdict"), Some("protocol 2 rejected"));
        // a large cell short by more than its lost cycles is rejected; short only by the lost ones is not yet
        let lost_large: Vec<_> = ok.iter().cloned().enumerate().filter(|(i, _)| *i != 2).map(|(_, r)| r).collect();
        assert_eq!(analyse(&lost_large, &plan, 3, 3).str("verdict"), Some("inconclusive (cycles lost to aborts)"));
        let mut short = ok.clone();
        short[2] = row("large", "ambient", "rejected", "rejected");
        assert_eq!(analyse(&short, &plan, 3, 3).str("verdict"), Some("protocol 2 rejected"));
    }
}
