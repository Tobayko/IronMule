//! Integration tests of the OPT1 cycle against the real binaries: activation and rollback, adapter
//! crash and hang, coordinator abort and resume, damaged checkpoint, altered journal, identity
//! change, wrong output, an unreachable goal, and the RL comparison end to end.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const IMOPT: &str = env!("CARGO_BIN_EXE_imopt");
/// The tests measure real CPU work: run them one at a time, or they disturb each other's timings
/// (on a 4-CPU host seven parallel tests made a real qualification fail its A/A noise gate).
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}
const CPUJOB: &str = env!("CARGO_BIN_EXE_imopt-cpujob");

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("imopt-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn goal(d: &Path, min_gain: f64) -> PathBuf {
    let p = d.join("goal.json");
    std::fs::write(&p, format!(r#"{{"schema":"imopt.goal.v1","objective":"test",
        "rule":{{"min_gain":{min_gain},"max_regression":0.5,"pairs":4,"family_alpha":0.2,"max_trials":2}},
        "budget":{{"max_tests":8,"max_seconds":120,"reference_runs":2}},
        "planner":{{"prior_sd":0.5,"noise_sd":0.05,"min_gain_per_s":0.0,"default_cost_s":0}}}}"#)).unwrap();
    p
}

fn imopt(d: &Path, goal: &Path, cmd: &str, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(IMOPT);
    c.args([cmd, "--state", d.join("st").to_str().unwrap(), "--goal", goal.to_str().unwrap(), "--timeout", "5", "--", CPUJOB])
        .env("IMOPT_CPUJOB_CONFIG", d.join("app.json")).env("IMOPT_CPUJOB_CHUNKS", "48").env("IMOPT_CPUJOB_CHUNK_KB", "64");
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().unwrap()
}

fn last_json(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).lines().last().unwrap_or("").to_string()
}

fn journal_kinds(d: &Path) -> Vec<String> {
    std::fs::read_to_string(d.join("st/journal.jsonl")).unwrap_or_default().lines()
        .filter_map(|l| ironmule_optimizer::json::parse(l).ok()).filter_map(|v| v.str("kind").map(String::from)).collect()
}

/// A deterministic Python adapter with its own evaluator (the IronMule adapter's shape).
fn python_adapter(d: &Path) -> PathBuf {
    let script = d.join("adapter.py");
    std::fs::write(&script, r#"
import json, os, sys
state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active.json")
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
cmd = args[0]
print("library chatter on stdout before the answer")
if cmd == "describe":
    out = {"name": "pyfake", "reference": "slow", "evaluator": "adapter", "identity": {"v": 1},
           "actions": [{"id": "slow", "settings": {}}, {"id": "fast", "settings": {}}, {"id": "odd", "class": "tested_only"}]}
elif cmd == "run":
    a = opt("--action")
    out = {"ok": True, "cost": {"slow": 2.0, "fast": 1.0, "odd": 0.5}[a], "correctness": "x" if a != "odd" else "y"}
elif cmd == "qualify":
    refuse = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refuse"))
    out = {"ok": True, "qualified": opt("--action") == "fast" and not refuse, "mean_ratio": 0.5, "reason": None}
elif cmd == "apply":
    prev = json.load(open(state)) if os.path.exists(state) else None
    json.dump({"action": opt("--action")}, open(state, "w"))
    out = {"ok": True, "previous": prev}
elif cmd == "revert":
    prev = json.loads(opt("--previous"))
    if prev is None:
        os.path.exists(state) and os.remove(state)
    else:
        json.dump(prev, open(state, "w"))
    out = {"ok": True}
elif cmd == "active":
    out = {"ok": True, "action": json.load(open(state))["action"] if os.path.exists(state) else None}
print(json.dumps(out))
"#).unwrap();
    script
}

fn multi_core() -> bool {
    std::thread::available_parallelism().map_or(1, |n| n.get()) >= 2
}

/// TEST1: a real adapter process whose measurements are controlled (fixed cost per action, a small
/// deterministic spread by seed, one action with a wrong output); the coordinator's own evaluator judges them.
fn controlled_adapter(d: &Path) -> PathBuf {
    let script = d.join("controlled.py");
    std::fs::write(&script, r#"
import json, os, sys
state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active.json")
args = sys.argv[1:]
scenario = args[0]
cmd = args[1]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
COSTS = {"accept": {"ref": 1.0, "near": 0.99, "fast": 0.5, "wrong": 0.4},
         "keep": {"ref": 1.0, "fast": 0.99, "near": 1.01, "wrong": 0.4},
         "sly": {"ref": 1.0, "sly": 0.5}, "late": {"ref": 1.0, "late": 0.5}}[scenario]
def bad(a, seed):  # wrong always; sly only in the qualification triplets (5000-8999); late only at the recheck (9000)
    return a == "wrong" or (a == "sly" and 5000 <= seed < 9000) or (a == "late" and seed == 9000)
if cmd == "describe":
    out = {"name": "controlled", "reference": "ref", "evaluator": "coordinator", "metric": "cost",
           "identity": {"app": "controlled", "scenario": scenario},
           "actions": [{"id": a, "settings": {"a": a}, "class": "auto"} for a in COSTS]}
elif cmd == "run":
    a, seed = opt("--action"), int(opt("--seed") or 0)
    spread = 1.0 + 0.004 * (seed % 3 - 1)
    out = {"ok": True, "cost": COSTS[a] * spread, "correctness": "bad" if bad(a, seed) else "good"}
elif cmd == "apply":
    prev = json.load(open(state)) if os.path.exists(state) else None
    json.dump({"action": opt("--action")}, open(state, "w"))
    out = {"ok": True, "previous": prev}
elif cmd == "revert":
    prev = json.loads(opt("--previous"))
    if prev is None:
        os.path.exists(state) and os.remove(state)
    else:
        json.dump(prev, open(state, "w"))
    out = {"ok": True}
elif cmd == "active":
    out = {"ok": True, "action": json.load(open(state))["action"] if os.path.exists(state) else None}
print(json.dumps(out))
"#).unwrap();
    script
}

fn controlled(d: &Path, goal: &Path, cmd: &str, scenario: &str) -> Output {
    Command::new(IMOPT).args([cmd, "--state", d.join("st").to_str().unwrap(), "--goal", goal.to_str().unwrap(), "--timeout", "20",
        "--", "python3", controlled_adapter(d).to_str().unwrap(), scenario]).output().unwrap()
}

fn bodies_of(d: &Path, kind: &str) -> Vec<ironmule_optimizer::json::Value> {
    std::fs::read_to_string(d.join("st/journal.jsonl")).unwrap_or_default().lines()
        .filter_map(|l| ironmule_optimizer::json::parse(l).ok())
        .filter(|v| v.str("kind") == Some(kind)).filter_map(|v| v.get("body").cloned()).collect()
}

/// TEST1, deterministic: the coordinator's unchanged evaluator qualifies a clearly faster candidate from
/// controlled measurements, activates it (the adapter reports it), does nothing more on a resumed cycle,
/// never qualifies the cheaper action with a wrong output, and rollback restores the default.
#[test]
fn the_coordinator_qualifies_activates_resumes_and_rolls_back_on_controlled_measurements() {
    let d = dir("controlled-accept");
    let g = goal(&d, 0.03);
    let o = controlled(&d, &g, "cycle", "accept");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(last_json(&o).contains("\"result\":\"activated\"") && last_json(&o).contains("\"fast\""), "{}", last_json(&o));
    let q = bodies_of(&d, "qualify_end");
    assert!(q.iter().any(|b| b.str("action") == Some("fast") && b.str("verdict").is_some_and(|v| v.starts_with("Qualified"))));
    assert!(q.iter().all(|b| b.str("action") != Some("wrong")), "a wrong output is never qualified");
    assert!(bodies_of(&d, "test").iter().any(|b| b.str("action") == Some("wrong") && b.boolean("same_output") == Some(false)));
    assert!(std::fs::read_to_string(d.join("active.json")).unwrap().contains("fast"), "the adapter holds the setting");
    let st = controlled(&d, &g, "status", "accept");
    assert!(st.status.success() && last_json(&st).contains("\"active\":{\"action\":\"fast\""), "{}", last_json(&st));
    let n = journal_kinds(&d).len();
    let again = controlled(&d, &g, "cycle", "accept");
    assert_eq!(last_json(&again), last_json(&o), "a resumed cycle reports the same outcome");
    assert_eq!(journal_kinds(&d).len(), n, "and measures nothing again");
    let rb = controlled(&d, &g, "rollback", "accept");
    assert!(last_json(&rb).contains("\"rolled_back\":true"), "{}", last_json(&rb));
    assert!(!d.join("active.json").exists(), "rollback restores the default");
}

/// TEST1, deterministic: candidates inside the minimum gain keep the reference, and the cheapest action,
/// whose output is wrong, is never a candidate.
#[test]
fn the_coordinator_keeps_the_reference_when_no_candidate_qualifies_on_controlled_measurements() {
    let d = dir("controlled-keep");
    let g = goal(&d, 0.03);
    let o = controlled(&d, &g, "cycle", "keep");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(last_json(&o).contains("reference_kept"), "{}", last_json(&o));
    assert!(bodies_of(&d, "activate").is_empty() && !d.join("active.json").exists());
    assert!(bodies_of(&d, "qualify_start").iter().all(|b| b.str("action") != Some("wrong")));
    assert!(bodies_of(&d, "test").iter().any(|b| b.str("action") == Some("wrong") && b.boolean("same_output") == Some(false)));
    // the best usable candidate did reach the evaluator, which rejected it
    assert!(bodies_of(&d, "qualify_end").iter().any(|b| b.str("action") == Some("fast")
        && b.str("verdict").is_some_and(|v| v.starts_with("Rejected"))), "fast was judged and rejected");
}

/// TEST1, deterministic: an output that turns wrong only during the qualification triplets is vetoed by the
/// evaluator's correctness check, and one that turns wrong only after activation fails the recheck, which
/// rolls the setting back.
#[test]
fn the_coordinator_vetoes_outputs_that_turn_wrong_in_qualification_or_after_activation() {
    let d = dir("controlled-sly");
    let g = goal(&d, 0.03);
    let o = controlled(&d, &g, "cycle", "sly");
    assert!(o.status.success() && last_json(&o).contains("reference_kept"), "{}", last_json(&o));
    assert!(bodies_of(&d, "qualify_end").iter().any(|b| b.str("action") == Some("sly")
        && b.str("verdict").is_some_and(|v| v.contains("correctness failed"))), "the triplets' digests are checked");
    assert!(bodies_of(&d, "activate").is_empty() && !d.join("active.json").exists());
    let d = dir("controlled-late");
    let o = controlled(&d, &g, "cycle", "late");
    assert!(o.status.success() && last_json(&o).contains("reference_kept"), "{}", last_json(&o));
    assert_eq!(bodies_of(&d, "activate").len(), 1, "it qualified and was applied once");
    assert!(bodies_of(&d, "recheck").iter().any(|b| b.boolean("passed") == Some(false)));
    assert!(bodies_of(&d, "rollback").iter().any(|b| b.str("result") == Some("reverted")));
    assert!(!d.join("active.json").exists(), "the failed recheck restores the default");
}

/// Hardware test (TEST1): a real cycle on real CPU work, whose protocol-1 qualification depends on this
/// machine's timing noise; it can fail under desktop load by design. Run on request on a quiet machine:
/// `cargo test --release --test coordinator -- --ignored`.
#[test]
#[ignore = "hardware timing test (TEST1): cargo test -- --ignored on a quiet machine"]
fn hardware_a_real_cycle_activates_a_qualified_thread_count_and_rollback_restores_the_default() {
    let _serial = serial();
    let d = dir("cycle");
    let g = goal(&d, 0.03);
    let o = imopt(&d, &g, "cycle", &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    if !multi_core() {
        assert!(last_json(&o).contains("reference_kept"));
        return;
    }
    assert!(last_json(&o).contains("\"result\":\"activated\""), "{}", last_json(&o));
    let app = std::fs::read_to_string(d.join("app.json")).unwrap();
    assert!(app.contains("threads"));
    // a second cycle of the same identity does not redo anything
    let n = journal_kinds(&d).len();
    let again = imopt(&d, &g, "cycle", &[]);
    assert_eq!(last_json(&again), last_json(&o));
    assert_eq!(journal_kinds(&d).len(), n);
    let rb = imopt(&d, &g, "rollback", &[]);
    assert!(last_json(&rb).contains("\"rolled_back\":true"));
    assert!(!d.join("app.json").exists(), "rollback must remove the applied setting");
    assert!(journal_kinds(&d).iter().any(|k| k == "rollback"));
}

#[test]
fn a_crashing_hanging_or_wrong_action_is_a_failed_test_never_a_candidate() {
    let _serial = serial();
    if !multi_core() {
        return;
    }
    let d = dir("faults");
    let g = goal(&d, 0.03);
    let o = imopt(&d, &g, "cycle", &[("IMOPT_FAULT", "crash:threads=2,hang:threads=3,wrong:threads=4")]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = std::fs::read_to_string(d.join("st/journal.jsonl")).unwrap();
    for (action, reason) in [("threads=2", "crashed"), ("threads=3", "timed out")] {
        assert!(text.lines().any(|l| l.contains("\"kind\":\"test\"") && l.contains(action) && l.contains(reason)), "{action}");
    }
    assert!(text.lines().any(|l| l.contains("\"kind\":\"test\"") && l.contains("threads=4") && l.contains("\"same_output\":false")));
    let out = last_json(&o);
    for bad in ["threads=2", "threads=3", "threads=4"] {
        assert!(!out.contains(bad), "{bad} must not be activated: {out}");
    }
}

#[test]
fn an_aborted_coordinator_resumes_from_its_journal_without_repeating_measurements() {
    let _serial = serial();
    let d = dir("abort");
    let g = goal(&d, 0.03);
    let mut child = Command::new(IMOPT)
        .args(["cycle", "--state", d.join("st").to_str().unwrap(), "--goal", g.to_str().unwrap(), "--", CPUJOB])
        .env("IMOPT_CPUJOB_CONFIG", d.join("app.json")).env("IMOPT_CPUJOB_CHUNKS", "48").env("IMOPT_CPUJOB_CHUNK_KB", "64")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let t0 = Instant::now();
    while journal_kinds(&d).iter().filter(|k| *k == "reference").count() < 2 && t0.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(2));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let before = journal_kinds(&d);
    let o = imopt(&d, &g, "cycle", &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let after = journal_kinds(&d);
    assert_eq!(after.iter().filter(|k| *k == "reference").count(), 2, "reference runs are not repeated after a resume");
    assert!(after.len() > before.len() && after.iter().any(|k| k == "outcome"));
}

#[test]
fn a_damaged_checkpoint_is_quarantined_and_an_altered_journal_is_refused() {
    let _serial = serial();
    let d = dir("damage");
    let g = goal(&d, 0.03);
    assert!(imopt(&d, &g, "cycle", &[]).status.success());
    let cp = d.join("st/planner.json");
    std::fs::write(&cp, std::fs::read_to_string(&cp).unwrap().replace("bytes_sha256", "bytes_sha257")).unwrap();
    assert!(imopt(&d, &g, "status", &[]).status.success());
    let rb = imopt(&d, &g, "rollback", &[]);
    assert!(rb.status.success());
    assert!(imopt(&d, &g, "cycle", &[]).status.success());
    assert!(journal_kinds(&d).iter().any(|k| k == "checkpoint_quarantined"));
    // one changed byte in the journal: everything stops, nothing is applied
    let j = d.join("st/journal.jsonl");
    let text = std::fs::read_to_string(&j).unwrap();
    std::fs::write(&j, text.replacen("\"kind\":\"reference\"", "\"kind\":\"reference \"", 1)).unwrap();
    let o = imopt(&d, &g, "cycle", &[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("journal"));
}

#[test]
fn a_setting_activated_under_another_identity_is_rolled_back_first() {
    let _serial = serial();
    let d = dir("identity");
    let script = python_adapter(&d);
    let run = |goal: &Path, cmd: &str| Command::new(IMOPT).args([cmd, "--state", d.join("st").to_str().unwrap(), "--goal",
        goal.to_str().unwrap(), "--", "python3", script.to_str().unwrap()]).output().unwrap();
    let g = goal(&d, 0.03);
    assert!(last_json(&run(&g, "cycle")).contains("\"action\":\"fast\""));
    assert!(d.join("active.json").exists());
    // a changed goal is a changed identity: the old activation is not qualified here and is reverted first;
    // the adapter now refuses to qualify, so nothing is activated again
    std::fs::write(d.join("refuse"), "").unwrap();
    let g2 = goal(&d, 0.04);
    let o = run(&g2, "cycle");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(journal_kinds(&d).iter().any(|k| k == "identity_rollback"));
    assert!(last_json(&o).contains("reference_kept"), "{}", last_json(&o));
    assert!(!d.join("active.json").exists(), "the foreign activation must be reverted");
}

#[test]
fn an_unreachable_goal_keeps_the_reference() {
    let _serial = serial();
    let d = dir("goal");
    let g = goal(&d, 0.95);
    let o = imopt(&d, &g, "cycle", &[]);
    assert!(o.status.success());
    assert!(last_json(&o).contains("reference_kept"), "{}", last_json(&o));
    assert!(!d.join("app.json").exists());
}

#[test]
fn the_comparison_runs_end_to_end_and_its_dream_choice_replays_identically() {
    let _serial = serial();
    let d = dir("compare");
    let st = d.join("st");
    let o = Command::new(IMOPT).args(["compare", "--state", st.to_str().unwrap(), "--worker", CPUJOB,
        "--budget", "2", "--holdout", "2", "--chunks", "40"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report = ironmule_optimizer::json::parse(&std::fs::read_to_string(st.join("compare-report.json")).unwrap()).unwrap();
    let h = report.get("holdout").unwrap();
    for s in ["sweep_best_fixed", "sweep_best_fixed#AA", "load_rule", "load_rule#AA", "planner", "qlearn", "qlearn_dream"] {
        assert_eq!(h.arr(s).map(|a| a.len()), Some(2), "{s}");
    }
    assert!(report.get("analysis").unwrap().get("rl_vs_best_simple").unwrap().str("verdict").is_some());
    let r1 = Command::new(IMOPT).args(["dream-replay", "--state", st.to_str().unwrap()]).output().unwrap();
    let r2 = Command::new(IMOPT).args(["dream-replay", "--state", st.to_str().unwrap()]).output().unwrap();
    assert!(r1.status.success());
    assert_eq!(r1.stdout, r2.stdout, "replay is deterministic");
    assert!(String::from_utf8_lossy(&r1.stdout).contains("\"matches_recorded_choice\":true"));
}

/// A Python adapter with its own evaluator (the IronMule adapter's shape): the coordinator calls its
/// `qualify`, activates only on its verdict, and rolls back through its `revert`.
#[test]
fn an_adapter_with_its_own_evaluator_is_qualified_by_it_across_the_language_boundary() {
    let _serial = serial();
    let d = dir("python");
    let script = python_adapter(&d);
    let g = goal(&d, 0.03);
    let run = |cmd: &str| Command::new(IMOPT).args([cmd, "--state", d.join("st").to_str().unwrap(), "--goal", g.to_str().unwrap(),
        "--", "python3", script.to_str().unwrap()]).output().unwrap();
    let o = run("cycle");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(last_json(&o).contains("\"action\":\"fast\""), "{}", last_json(&o));
    assert!(d.join("active.json").exists());
    let text = std::fs::read_to_string(d.join("st/journal.jsonl")).unwrap();
    assert!(text.contains("\"evaluator\":\"adapter\""));
    assert!(text.lines().any(|l| l.contains("\"kind\":\"test\"") && l.contains("\"action\":\"odd\"") && l.contains("\"usable\":false")),
            "a tested_only action with different output is measured but never usable");
    assert!(last_json(&run("rollback")).contains("true"));
    assert!(!d.join("active.json").exists());
}

/// Review finding: a crash after `activate` and before `outcome` must not apply again (which would
/// record the candidate as its own previous state); the resume rechecks, and rollback restores the
/// real default. Also: rollback with a lost envelope uses the journal, and an outcome whose activation
/// vanished outside the coordinator is invalidated rather than reported as standing.
#[test]
fn a_crash_between_activation_and_outcome_resumes_without_a_second_apply() {
    let _serial = serial();
    let d = dir("activate-crash");
    let script = python_adapter(&d);
    let g = goal(&d, 0.03);
    let run = |cmd: &str| Command::new(IMOPT).args([cmd, "--state", d.join("st").to_str().unwrap(), "--goal",
        g.to_str().unwrap(), "--", "python3", script.to_str().unwrap()]).output().unwrap();
    assert!(last_json(&run("cycle")).contains("\"action\":\"fast\""));
    // Cut the journal back to its `activate` record, as if the coordinator died right after it.
    let j = d.join("st/journal.jsonl");
    let lines: Vec<String> = std::fs::read_to_string(&j).unwrap().lines().map(String::from).collect();
    let k = lines.iter().position(|l| l.contains("\"kind\":\"activate\"")).unwrap();
    std::fs::write(&j, lines[..=k].join("\n") + "\n").unwrap();
    let rec = ironmule_optimizer::json::parse(&lines[k]).unwrap();
    std::fs::write(d.join("st/journal.jsonl.head"), format!("{{\"hash\":\"{}\",\"seq\":{}}}\n", rec.str("hash").unwrap(), k)).unwrap();
    let applies_before = std::fs::read_to_string(&j).unwrap().matches("\"kind\":\"activate\"").count();
    let o = run("cycle");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(last_json(&o).contains("\"action\":\"fast\""));
    let text = std::fs::read_to_string(&j).unwrap();
    assert_eq!(text.matches("\"kind\":\"activate\"").count(), applies_before, "no second apply on resume");
    assert!(text.contains("\"kind\":\"recheck\"") && text.contains("\"kind\":\"outcome\""));
    // the envelope is lost too: rollback still restores the real previous state (none) from the journal
    std::fs::remove_file(d.join("st/active.json")).unwrap();
    assert!(last_json(&run("rollback")).contains("true"));
    assert!(!d.join("active.json").exists(), "rollback must restore the default, not the candidate");
    // activate again, then remove the activation behind the coordinator's back: the outcome is invalidated
    assert!(last_json(&run("cycle")).contains("\"action\":\"fast\""));
    std::fs::remove_file(d.join("active.json")).unwrap();
    assert!(run("cycle").status.success());
    assert!(std::fs::read_to_string(&j).unwrap().contains("\"kind\":\"outcome_invalidated\""));
}

fn campaign(st: &Path, extra: &[&str], fault: Option<&str>) -> Output {
    let mut c = Command::new(IMOPT);
    c.args(["campaign", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40", "--threads", "1,2",
            "--baseline-threads", "1", "--budget", "3", "--holdout", "2", "--rounds", "2"]).args(extra);
    if let Some(f) = fault {
        c.env("IMOPT_FAULT", f);
    }
    c.output().unwrap()
}

fn bodies(st: &Path, kind: &str) -> Vec<ironmule_optimizer::json::Value> {
    std::fs::read_to_string(st.join("journal.jsonl")).unwrap_or_default().lines()
        .filter_map(|l| ironmule_optimizer::json::parse(l).ok())
        .filter(|v| v.str("kind") == Some(kind)).filter_map(|v| v.get("body").cloned()).collect()
}

/// OPT3: a campaign killed in training and in a holdout resumes from its journal: the aborted training
/// attempt keeps its budget charge and is never a measurement, the interrupted holdout schedule is
/// burned (none of its arms runs again), the learner is rebuilt and checked against the digest the
/// online learner journaled, and a finished campaign does nothing when started again.
#[test]
fn a_campaign_resumes_after_aborts_without_resetting_budget_or_holdout() {
    let _serial = serial();
    let d = dir("campaign-abort");
    let st = d.join("st");
    let o = campaign(&st, &[], Some("campaign_abort:1"));
    assert_eq!(o.status.code(), Some(9), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!((bodies(&st, "attempt").len(), bodies(&st, "episode").len()), (2, 1));
    let o = campaign(&st, &["--status"], None);
    assert!(o.status.success() && last_json(&o).contains("\"attempts\":2"));
    // the round-1 holdout begins after the charged budget: kill its third attempt
    let o = campaign(&st, &[], Some("campaign_abort:4"));
    assert_eq!(o.status.code(), Some(9));
    let aborted = bodies(&st, "attempt_aborted");
    assert_eq!(aborted.len(), 1, "the first interruption is recorded once");
    assert_eq!((aborted[0].num("attempt"), aborted[0].num("charged_s")), (Some(1.0), Some(30.0)));
    let o = campaign(&st, &[], None);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let aborted = bodies(&st, "attempt_aborted");
    assert_eq!(aborted.len(), 2);
    let burned = aborted[1].num("seed").unwrap();
    assert_eq!(aborted[1].str("phase"), Some("holdout"));
    let after: Vec<_> = bodies(&st, "episode").into_iter().filter(|b| b.num("seed") == Some(burned)).collect();
    assert!(after.len() < 4, "a burned schedule never completes: {} arms", after.len());
    assert!(bodies(&st, "attempt").iter().filter(|b| b.num("seed") == Some(burned)).count() <= 3, "no arm retried");
    // aborted attempts are never measurements; every training episode is one attempt
    let attempts: Vec<f64> = bodies(&st, "episode").iter().filter_map(|b| b.num("attempt")).collect();
    assert!(!attempts.contains(&1.0) && !attempts.contains(&4.0));
    let r1 = bodies(&st, "round_decision");
    assert_eq!(r1.len(), 2);
    assert_eq!(r1[0].str("verdict"), Some("inconclusive (holdout schedules lost to aborts)"));
    let status = last_json(&o);
    assert!(status.contains("\"done\":true"), "{status}");
    let n = bodies(&st, "attempt").len();
    let again = campaign(&st, &[], None);
    assert!(again.status.success());
    assert_eq!(bodies(&st, "attempt").len(), n, "a finished campaign runs nothing again");
    // the same state with another plan or another identity is refused, never mixed in
    let other = campaign(&st, &["--seed", "2"], None);
    assert!(!other.status.success() && String::from_utf8_lossy(&other.stderr).contains("another plan"));
    let foreign = Command::new(IMOPT).args(["campaign", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "41",
        "--threads", "1,2", "--baseline-threads", "1", "--budget", "3", "--holdout", "2"]).output().unwrap();
    assert!(!foreign.status.success() && String::from_utf8_lossy(&foreign.stderr).contains("do not transfer"));
}

/// OPT3 acceptance: round 2 learns on top of round 1's real training episodes (policy version and
/// update count carry over), holdout episodes never enter the learner, round 2's holdout schedules are
/// new, and the replay selection between the rounds sees training trees only.
#[test]
fn experience_from_round_one_enters_round_two_without_holdout_leakage() {
    let _serial = serial();
    let d = dir("campaign-rounds");
    let st = d.join("st");
    let o = campaign(&st, &[], None);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let train: Vec<_> = bodies(&st, "episode").into_iter().filter(|b| b.str("phase") == Some("train")).collect();
    let r1: Vec<_> = train.iter().filter(|b| b.num("round") == Some(1.0)).collect();
    let r1_updates: f64 = r1.iter().map(|b| b.arr("transitions").unwrap().len() as f64).sum();
    let first_r2 = bodies(&st, "attempt").into_iter().find(|b| b.str("phase") == Some("train") && b.num("round") == Some(2.0)).unwrap();
    assert!(!r1.is_empty());
    assert_eq!(first_r2.num("policy_version"), Some(r1.len() as f64), "round 2 starts from round 1's episodes");
    assert_eq!(first_r2.num("updates_before"), Some(r1_updates));
    let all_train: f64 = train.iter().map(|b| b.arr("transitions").unwrap().len() as f64).sum();
    let status = ironmule_optimizer::json::parse(&last_json(&o)).unwrap();
    assert_eq!(status.num("learned_updates"), Some(all_train), "holdout episodes never update the learner");
    let holdout_seeds = |r: f64| -> std::collections::BTreeSet<u64> {
        bodies(&st, "episode").iter().filter(|b| b.str("phase") == Some("holdout") && b.num("round") == Some(r))
            .map(|b| b.num("seed").unwrap() as u64).collect()
    };
    let (h1, h2) = (holdout_seeds(1.0), holdout_seeds(2.0));
    assert!(!h1.is_empty() && h1.is_disjoint(&h2));
    assert!(train.iter().all(|b| !h1.contains(&(b.num("seed").unwrap() as u64)) && !h2.contains(&(b.num("seed").unwrap() as u64))));
    let select = bodies(&st, "select");
    assert_eq!(select.len(), 1, "one replay selection between the two rounds");
    let train_slices: f64 = train.iter().filter(|b| b.num("round") == Some(1.0)).map(|b| b.arr("transitions").unwrap().len() as f64).sum();
    assert_eq!(select[0].num("dev_transitions").unwrap() + select[0].num("val_transitions").unwrap(), train_slices,
               "replay sees round 1's training slices and nothing else");
    // --status works without the plan's flags; the fresh comparison refuses a campaign's state
    let s = Command::new(IMOPT).args(["campaign", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40",
        "--threads", "1,2", "--status"]).output().unwrap();
    assert!(s.status.success() && last_json(&s).contains("\"done\":true"), "{}", String::from_utf8_lossy(&s.stderr));
    let c = Command::new(IMOPT).args(["compare", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40"]).output().unwrap();
    assert!(!c.status.success() && String::from_utf8_lossy(&c.stderr).contains("learning campaign"));
}

fn opt2(st: &Path, args: &[&str], fault: Option<&str>) -> Output {
    let mut c = Command::new(IMOPT);
    c.args(["campaign", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40", "--threads", "1,2",
            "--baseline-threads", "1"]).args(args);
    if let Some(f) = fault {
        c.env("IMOPT_FAULT", f);
    }
    c.output().unwrap()
}

/// OPT2: a warm, dense campaign sweeps every action first (its sweep charged to the budget, a killed sweep
/// attempt charged and never rerun), starts its learner with the sweep's best action greedy, trains on the
/// dense reward while every return stays the sparse one, and holds out exactly the declared seeds. A duel
/// judges two finished campaigns' final learners on the same schedules and runs nothing when started again.
#[test]
fn an_opt2_campaign_sweeps_then_learns_dense_and_a_duel_compares_two_learners() {
    use ironmule_optimizer::episode::dense_reward;
    let _serial = serial();
    let d = dir("opt2");
    let st = d.join("warm");
    let plan = ["--rounds", "1", "--budget", "33", "--holdout", "2", "--holdout-seed0", "5000", "--reward", "dense",
                "--start", "warm", "--no-dream"];
    let o = opt2(&st, &plan, Some("campaign_abort:0"));
    assert_eq!(o.status.code(), Some(9), "{}", String::from_utf8_lossy(&o.stderr));
    let o = opt2(&st, &plan, None);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let attempts = bodies(&st, "attempt");
    let aborted = bodies(&st, "attempt_aborted");
    assert_eq!(aborted.len(), 1);
    assert_eq!((aborted[0].str("phase"), aborted[0].num("charged_s")), (Some("sweep"), Some(30.0)));
    let episodes = bodies(&st, "episode");
    let sweep: Vec<_> = episodes.iter().filter(|b| b.str("phase") == Some("sweep")).collect();
    assert!(sweep.iter().all(|b| b.num("attempt") != Some(0.0)), "the killed sweep attempt is no measurement");
    let mean = |a: f64| {
        let r: Vec<f64> = sweep.iter().filter(|b| b.num("action") == Some(a)).map(|b| b.num("return").unwrap()).collect();
        assert!(!r.is_empty(), "every action swept");
        r.iter().sum::<f64>() / r.len() as f64
    };
    let prior = vec![mean(0.0), mean(1.0)];
    let last_sweep = sweep.iter().map(|b| b.num("attempt").unwrap()).fold(0.0, f64::max);
    let train: Vec<_> = attempts.iter().filter(|b| b.str("phase") == Some("train")).collect();
    assert!(!train.is_empty() && train.iter().all(|b| b.num("attempt").unwrap() > last_sweep), "training follows the sweep");
    assert!(train.iter().all(|b| b.arr("prior").map(|p| p.iter().filter_map(|x| x.as_f64()).collect::<Vec<_>>()) == Some(prior.clone())
                              && b.boolean("dense") == Some(true)), "the learner starts from the sweep's mean returns");
    let plan_body = &bodies(&st, "campaign_plan")[0];
    let targets: Vec<f64> = plan_body.arr("foreground_targets_ms").unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::INFINITY)).collect();
    for e in episodes.iter().filter(|b| b.str("phase") == Some("train")) {
        let tr = e.arr("transitions").unwrap();
        let sparse: f64 = tr.iter().map(|t| if t.boolean("violated") == Some(true) { -5.0 } else { -1.0 }).sum();
        assert_eq!(e.num("return"), Some(sparse), "the return is the sparse one");
        for t in tr {
            let want = dense_reward(t.num("p95_ms"), targets[t.num("level").unwrap() as usize], 4.0);
            assert!((t.num("reward").unwrap() - want).abs() < 1e-9, "the learner trains on the dense reward");
        }
    }
    let seeds: std::collections::BTreeSet<u64> = episodes.iter().filter(|b| b.str("phase") == Some("holdout"))
        .map(|b| b.num("seed").unwrap() as u64).collect();
    assert_eq!(seeds, [5001, 5002].into_iter().collect());
    // phase D: the same variant with and without dream, then both final learners on the same schedules
    let base_dir = d.join("nodream");
    let run = |name: &str, dream: bool| {
        let st = d.join(name);
        let mut a = vec!["--rounds", "2", "--budget", "2", "--holdout", "0", "--seed", "7"];
        if !dream {
            a.push("--no-dream");
        } else {
            a.extend(["--targets-from", base_dir.to_str().unwrap()]);
        }
        let o = opt2(&st, &a, None);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        st
    };
    let (base, cand) = (run("nodream", false), run("dream", true));
    assert_eq!(bodies(&cand, "select").len(), 1);
    let targets = |st: &Path| bodies(st, "campaign_plan")[0].get("foreground_targets_ms").unwrap().dump();
    assert_eq!(targets(&base), targets(&cand), "the dream campaign trains against the base's targets");
    // a candidate with its own targets is refused by the duel
    let own = d.join("own");
    assert!(opt2(&own, &["--rounds", "2", "--budget", "2", "--holdout", "0", "--seed", "7"], None).status.success());
    let duel = |st: &Path, b: &Path| Command::new(IMOPT).args(["duel", "--state", st.to_str().unwrap(), "--worker", CPUJOB,
        "--chunks", "40", "--threads", "1,2", "--base", b.to_str().unwrap(), "--candidate", cand.to_str().unwrap(),
        "--seeds", "6001-6002"]).output().unwrap();
    let ds = d.join("duel");
    let o = duel(&ds, &base);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let decision = bodies(&ds, "duel_decision");
    assert_eq!((decision.len(), decision[0].num("schedules"), decision[0].num("planned")), (1, Some(2.0), Some(2.0)));
    let arms: Vec<_> = bodies(&ds, "episode").iter().map(|b| (b.num("seed").unwrap() as u64, b.str("arm").unwrap().to_string())).collect();
    assert_eq!(arms.len(), 8);
    assert_eq!(arms[0], (6001, "L".to_string()), "odd schedules start with the candidate");
    let n = bodies(&ds, "attempt").len();
    assert!(duel(&ds, &base).status.success());
    assert_eq!(bodies(&ds, "attempt").len(), n, "a finished duel runs nothing again");
    let o = duel(&d.join("duel2"), &d.join("missing"));
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("no campaign"));
    let o = Command::new(IMOPT).args(["duel", "--state", d.join("duel3").to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40",
        "--threads", "1,2", "--base", base.to_str().unwrap(), "--candidate", own.to_str().unwrap(), "--seeds", "6001-6002"])
        .output().unwrap();
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("different foreground targets"));
    // fixed holdout seeds belong to one round
    let o = opt2(&d.join("tworounds"), &["--rounds", "2", "--holdout", "2", "--holdout-seed0", "5000"], None);
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("one round"));
}

/// Activation happens once even if the coordinator dies between envelope and journal; the product mode
/// runs it and rollback restores the reference. The acceptance is forced, so this needs the
/// `test-faults` feature (a release build cannot force a verdict):
/// `cargo test --offline --release --features test-faults`.
#[cfg(feature = "test-faults")]
#[test]
fn a_campaign_activates_once_across_a_crash_and_rolls_back() {
    let _serial = serial();
    let d = dir("campaign-activate");
    let st = d.join("st");
    let o = campaign(&st, &[], Some("campaign_accept,campaign_crash_activate"));
    assert_eq!(o.status.code(), Some(9), "{}", String::from_utf8_lossy(&o.stderr));
    let o = campaign(&st, &[], Some("campaign_accept"));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!((bodies(&st, "activate_intent").len(), bodies(&st, "activate").len()), (1, 1));
    let env = ironmule_optimizer::json::parse(&std::fs::read_to_string(st.join("concurrency-active.json")).unwrap()).unwrap();
    assert_eq!(env.get("payload").unwrap().get("previous"), Some(&ironmule_optimizer::json::Value::Null));
    let m = |seed: &str| Command::new(IMOPT).args(["managed", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--chunks", "40",
        "--threads", "1,2", "--seed", seed]).output().unwrap();
    assert!(last_json(&m("7")).contains("\"policy_source\":\"active policy\""));
    assert!(Command::new(IMOPT).args(["managed-rollback", "--state", st.to_str().unwrap()]).output().unwrap().status.success());
    assert!(last_json(&m("8")).contains("\"policy_source\":\"reference (nothing active)\""));
}

/// OPT4b harness end to end on a tiny job: the control matrix, one cycle per cell, every cycle with all
/// its triplets judged by both protocols; an interrupted cycle is recorded once and never rerun.
#[test]
fn the_rule_study_judges_both_protocols_on_the_same_triplets_and_never_reruns_a_cycle() {
    let _serial = serial();
    let d = dir("rulecheck");
    let st = d.join("st");
    let g = goal(&d, 0.03);
    let run = |fault: Option<&str>| {
        let mut c = Command::new(IMOPT);
        c.args(["rulecheck", "--state", st.to_str().unwrap(), "--worker", CPUJOB, "--goal", g.to_str().unwrap(),
                "--cycles", "1", "--control"]).env("IMOPT_CPUJOB_CHUNKS", "24").env("IMOPT_CPUJOB_CHUNK_KB", "64");
        if let Some(f) = fault {
            c.env("IMOPT_FAULT", f);
        }
        c.output().unwrap()
    };
    assert_eq!(run(Some("rulecheck_abort:1")).status.code(), Some(9));
    let o = run(None);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let cycles = bodies(&st, "cycle");
    assert_eq!(cycles.len(), 3, "four planned, one interrupted and never rerun");
    assert_eq!(bodies(&st, "cycle_aborted").len(), 1);
    assert!(cycles.iter().all(|c| c.num("index") != Some(1.0)));
    for c in &cycles {
        assert_eq!(c.arr("triplets").unwrap().len(), 4, "every triplet collected, no early stop");
        for t in c.arr("triplets").unwrap() {
            let runs = t.arr("runs").unwrap();
            assert_eq!(runs.len(), 3, "each run is journaled with its own metrics");
            assert!(runs.iter().all(|r| r.get("metrics").and_then(|m| m.num("pad")) == Some(0.0)), "no pad outside the sub arm");
        }
        assert!(c.num("hog_spawn_failures") == Some(0.0));
        let j = c.get("judged").unwrap();
        assert_eq!(j.num("triplets"), Some(4.0));
        assert!(j.get("p1").and_then(|v| v.str("verdict")).is_some() && j.get("p2").and_then(|v| v.str("verdict")).is_some());
    }
    let report = ironmule_optimizer::json::parse(&last_json(&o)).unwrap();
    // a lost cycle is counted and can never pass; on this tiny job the large arm may also decide "rejected"
    assert_eq!(report.num("cycles_lost"), Some(1.0));
    assert_ne!(report.str("verdict"), Some("protocol 2 passes"));
    // an inherited pad or fault would change every arm: refused before anything runs
    let fresh = d.join("st2");
    let refused = Command::new(IMOPT).args(["rulecheck", "--state", fresh.to_str().unwrap(), "--worker", CPUJOB, "--goal",
        g.to_str().unwrap(), "--cycles", "1", "--control"]).env("IMOPT_CPUJOB_PAD", "0.5").output().unwrap();
    assert!(!refused.status.success() && String::from_utf8_lossy(&refused.stderr).contains("IMOPT_CPUJOB_PAD"));
    // a finished study runs nothing again
    assert!(run(None).status.success());
    assert_eq!(bodies(&st, "cycle").len(), 3);
}

/// Injected load never outlives its parent (a killed harness must not leave a hog burning CPU).
#[cfg(unix)]
#[test]
fn an_orphaned_hog_exits_by_itself() {
    let _serial = serial();
    let out = Command::new("sh").args(["-c", &format!("{CPUJOB} hog 1 $$ >/dev/null 2>&1 & echo $!")])
        .env("IMOPT_CPUJOB_CHUNKS", "8").env("IMOPT_CPUJOB_CHUNK_KB", "64").output().unwrap();
    let pid = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let alive = || Command::new("kill").args(["-0", &pid]).status().map_or(false, |s| s.success());
    let t0 = Instant::now();
    while alive() && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!alive(), "hog {pid} still running after its parent exited");
}
