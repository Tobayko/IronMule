//! OPT3-D (c): the coordinator's own measurement processes end without manual cleanup. Deterministic
//! contract tests against real processes (no timing claim beyond the fixed deadlines): a fixture adapter
//! records its own and its descendants' pids; every check is that those exact processes are gone (a pid
//! counts as alive only while it still runs the fixture's command, so a reused pid is never mistaken).
#![cfg(unix)]

use ironmule_optimizer::adapter::{Adapter, CallError};
use ironmule_optimizer::proc::{DEADLINE, GRACE};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const IMOPT: &str = env!("CARGO_BIN_EXE_imopt");
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

const FIXTURE: &str = r#"
import json, os, signal, subprocess, sys, time
d = os.environ["FIXTURE_DIR"]
mode = os.environ.get("FIXTURE_MODE", "ok")
cmd = sys.argv[1]
def note(role, pid):
    with open(os.path.join(d, f"{role}-{pid}"), "w") as f:
        f.write(repr(time.time()))
note(cmd, os.getpid())
def descendant(ignore_term):
    code = ("import signal, time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "time.sleep(300)")
    p = subprocess.Popen([sys.executable, "-c", code, "imopt-fixture-descendant"])  # inherits stdout
    note(cmd + "-child", p.pid)
if cmd == "describe":
    if mode == "slow_describe":
        descendant(False)
        time.sleep(300)
    out = {"name": "fixture", "reference": "a", "evaluator": "coordinator", "identity": {"v": 1},
           "actions": [{"id": "a", "settings": {}}, {"id": "b", "settings": {}}]}
elif cmd == "run":
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "graceful":
        def on_term(*_):
            note("term", os.getpid())  # proof that SIGTERM arrived and was handled
            sys.exit(0)
        signal.signal(signal.SIGTERM, on_term)
        time.sleep(300)
    if mode in ("leak", "hang", "stubborn"):
        descendant(mode == "stubborn")
    if mode in ("hang", "stubborn"):
        time.sleep(300)
    out = {"ok": True, "cost": 1.0, "correctness": "x"}
else:
    out = {"ok": True, "action": None, "previous": None}
print(json.dumps(out), flush=True)
"#;

struct Fx {
    dir: PathBuf,
    pids: PathBuf,
    script: PathBuf,
}

fn fixture(name: &str) -> Fx {
    let dir = std::env::temp_dir().join(format!("imopt-proc-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let pids = dir.join("pids");
    std::fs::create_dir_all(&pids).unwrap();
    let script = dir.join("imopt-fixture.py");
    std::fs::write(&script, FIXTURE).unwrap();
    std::fs::write(dir.join("goal.json"), r#"{"schema":"imopt.goal.v1","objective":"test",
        "rule":{"min_gain":0.03,"max_regression":0.5,"pairs":2,"family_alpha":0.2,"max_trials":1},
        "budget":{"max_tests":2,"max_seconds":60,"reference_runs":1},
        "planner":{"prior_sd":0.5,"noise_sd":0.05,"min_gain_per_s":0.0,"default_cost_s":0}}"#).unwrap();
    Fx { dir, pids, script }
}

/// Pids the fixture recorded whose file name starts with `role`.
fn recorded(fx: &Fx, role: &str) -> Vec<u32> {
    std::fs::read_dir(&fx.pids).unwrap().filter_map(|e| e.ok()).filter_map(|e| {
        let name = e.file_name().to_string_lossy().to_string();
        name.strip_prefix(&format!("{role}-")).and_then(|p| p.parse().ok())
    }).collect()
}

/// Alive only if the pid exists and still runs one of the fixture's commands.
fn alive(pid: u32) -> bool {
    let out = Command::new("ps").args(["-p", &pid.to_string(), "-o", "stat=,command="]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    !text.trim().is_empty() && !text.trim_start().starts_with('Z')
        && (text.contains("imopt-fixture") || text.contains("imopt-fixture-descendant"))
}

fn wait_for(what: &str, limit: Duration, mut cond: impl FnMut() -> bool) -> Duration {
    let t0 = Instant::now();
    while !cond() {
        assert!(t0.elapsed() < limit, "{what} not reached within {limit:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
    t0.elapsed()
}

fn adapter(fx: &Fx, timeout: Duration) -> Adapter {
    Adapter::new(vec!["python3".into(), fx.script.display().to_string()], timeout)
}

fn supervised(fx: &Fx, timeout: Duration) -> Adapter {
    adapter(fx, timeout).supervised(PathBuf::from(IMOPT), fx.dir.join("measure.lease")).unwrap()
}

fn imopt(fx: &Fx, cmd: &str, mode: &str) -> Child {
    Command::new(IMOPT).args([cmd, "--state", fx.dir.join("st").to_str().unwrap(), "--goal",
        fx.dir.join("goal.json").to_str().unwrap(), "--", "python3", fx.script.to_str().unwrap()])
        .env("FIXTURE_DIR", &fx.pids).env("FIXTURE_MODE", mode).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap()
}

/// All processes below `root` right now (an observation for the test, never used to end anything).
fn tree(root: u32) -> Vec<(u32, String)> {
    let out = String::from_utf8_lossy(&Command::new("ps").args(["-Ao", "pid=,ppid=,command="]).output().unwrap().stdout).to_string();
    let rows: Vec<(u32, u32, String)> = out.lines().filter_map(|l| {
        let mut it = l.split_whitespace();
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.collect::<Vec<_>>().join(" ")))
    }).collect();
    let mut found = vec![root];
    let mut i = 0;
    while i < found.len() {
        for (pid, ppid, _) in &rows {
            if *ppid == found[i] && !found.contains(pid) {
                found.push(*pid);
            }
        }
        i += 1;
    }
    rows.into_iter().filter(|(pid, _, _)| *pid != root && found.contains(pid)).map(|(pid, _, cmd)| (pid, cmd)).collect()
}

/// Still running the same command (a reused pid with another command does not count).
fn still(pid: u32, cmd: &str) -> bool {
    let out = Command::new("ps").args(["-p", &pid.to_string(), "-o", "stat=,command="]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    !text.is_empty() && !text.starts_with('Z')
        && text.split_once(char::is_whitespace).map(|x| x.1.split_whitespace().collect::<Vec<_>>().join(" "))
            == Some(cmd.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn sigkill(child: &mut Child) {
    Command::new("kill").args(["-KILL", &child.id().to_string()]).status().unwrap();
    let _ = child.wait();
}

/// Without supervision a call hangs while a leaked descendant holds the adapter's stdout; under
/// supervision it returns at once and the descendant is ended.
#[test]
fn a_normal_call_returns_although_a_descendant_holds_stdout_and_the_descendant_is_ended() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("leak");
    std::env::set_var("FIXTURE_DIR", &fx.pids);
    std::env::set_var("FIXTURE_MODE", "leak");
    // unsupervised: still blocked after 3 s (the bug), released only when the test ends the descendant
    let direct = adapter(&fx, Duration::from_secs(60));
    let handle = std::thread::spawn(move || direct.call(&["run", "--action", "a", "--seed", "1"]));
    std::thread::sleep(Duration::from_secs(3));
    assert!(!handle.is_finished(), "an unsupervised call should block on the leaked stdout");
    for pid in recorded(&fx, "run-child") {
        Command::new("kill").args(["-KILL", &pid.to_string()]).status().unwrap();
    }
    assert!(handle.join().unwrap().is_ok());
    // supervised: returns promptly, the leaked descendant is gone
    let _ = std::fs::remove_dir_all(&fx.pids);
    std::fs::create_dir_all(&fx.pids).unwrap();
    let t0 = Instant::now();
    let v = supervised(&fx, Duration::from_secs(60)).call(&["run", "--action", "a", "--seed", "1"]).unwrap();
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    assert_eq!(v.num("cost"), Some(1.0));
    let child = recorded(&fx, "run-child");
    assert_eq!(child.len(), 1);
    assert!(!alive(child[0]), "the leaked descendant was ended before the call returned");
}

/// A timeout ends the adapter and its descendant, even ones that ignore SIGTERM.
#[test]
fn a_timeout_ends_the_adapter_and_its_descendants() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("timeout");
    std::env::set_var("FIXTURE_DIR", &fx.pids);
    std::env::set_var("FIXTURE_MODE", "stubborn");
    let t0 = Instant::now();
    let r = supervised(&fx, Duration::from_secs(1)).call(&["run", "--action", "a", "--seed", "1"]);
    assert_eq!(r.unwrap_err(), CallError::Timeout);
    assert!(t0.elapsed() < Duration::from_secs(1) + DEADLINE, "{:?}", t0.elapsed());
    assert_eq!((recorded(&fx, "run").len(), recorded(&fx, "run-child").len()), (1, 1), "both had started (positive control)");
    for pid in recorded(&fx, "run").into_iter().chain(recorded(&fx, "run-child")) {
        assert!(!alive(pid), "pid {pid} survived the timeout");
    }
}

/// SIGKILL of the coordinator while a test runs: its adapter (ignoring SIGTERM) and that adapter's
/// descendant end within the deadline, with nobody else ending them. Same-named processes that are
/// not its own (the fixture started outside the coordinator) stay untouched.
#[test]
fn a_killed_coordinator_leaves_no_measurement_process_and_touches_no_foreign_one() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("kill-run");
    let foreign_dir = fx.dir.join("foreign");
    std::fs::create_dir_all(&foreign_dir).unwrap();
    let mut foreign = Command::new("python3").args([fx.script.to_str().unwrap(), "run"]).env("FIXTURE_DIR", &foreign_dir)
        .env("FIXTURE_MODE", "hang").stdout(Stdio::null()).spawn().unwrap();
    let mut c = imopt(&fx, "cycle", "stubborn");
    wait_for("the adapter's first run", Duration::from_secs(30), || !recorded(&fx, "run-child").is_empty());
    let own: Vec<u32> = recorded(&fx, "run").into_iter().chain(recorded(&fx, "run-child")).collect();
    assert!(own.iter().all(|p| alive(*p)));
    sigkill(&mut c);
    let took = wait_for("own processes gone", DEADLINE, || own.iter().all(|p| !alive(*p)));
    assert!(took >= GRACE.saturating_sub(Duration::from_millis(500)), "SIGTERM was ignored, so SIGKILL came after the grace");
    let foreign_pids: Vec<u32> = std::fs::read_dir(&foreign_dir).unwrap().filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().rsplit('-').next().and_then(|p| p.parse().ok())).collect();
    assert_eq!(foreign_pids.len(), 2);
    assert!(foreign_pids.iter().all(|p| alive(*p)), "a same-named foreign process was touched");
    for p in foreign_pids {
        Command::new("kill").args(["-KILL", &p.to_string()]).status().unwrap();
    }
    let _ = foreign.wait();
}

/// SIGKILL while the coordinator is still starting (its first adapter call, `describe`, under way), and
/// SIGKILL right after launch: nothing of it survives the deadline.
#[test]
fn a_coordinator_killed_during_start_leaves_nothing() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("kill-start");
    let mut c = imopt(&fx, "status", "slow_describe");
    wait_for("describe under way", Duration::from_secs(30), || !recorded(&fx, "describe-child").is_empty());
    let own: Vec<u32> = recorded(&fx, "describe").into_iter().chain(recorded(&fx, "describe-child")).collect();
    sigkill(&mut c);
    wait_for("own processes gone", DEADLINE, || own.iter().all(|p| !alive(*p)));
    // killed as soon as its first supervisor exists, before or while the adapter starts: everything the
    // coordinator had started at that moment is gone within the deadline (non-vacuous: at least the supervisor)
    let fx = fixture("kill-launch");
    let mut c = imopt(&fx, "status", "slow_describe");
    let mut seen = Vec::new();
    wait_for("a supervisor", Duration::from_secs(30), || {
        seen = tree(c.id());
        seen.iter().any(|(_, cmd)| cmd.contains(" supervise "))
    });
    sigkill(&mut c);
    wait_for("everything it had started gone", DEADLINE, || {
        seen.iter().all(|(p, cmd)| !still(*p, cmd)) && recorded(&fx, "describe").into_iter()
            .chain(recorded(&fx, "describe-child")).all(|p| !alive(p))
    });
}

/// A resumed coordinator starts no adapter call before the killed one's processes are gone; with the
/// lease held by a supervisor whose group is not gone, it refuses after its gate wait.
#[test]
fn a_resume_waits_for_the_old_processes_and_refuses_while_cleanup_is_open() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("resume");
    let mut c = imopt(&fx, "cycle", "stubborn");
    wait_for("the adapter's first run", Duration::from_secs(30), || !recorded(&fx, "run-child").is_empty());
    let old: Vec<u32> = recorded(&fx, "run").into_iter().chain(recorded(&fx, "run-child")).collect();
    let describes_before = recorded(&fx, "describe").len();
    sigkill(&mut c);
    let resumed = imopt(&fx, "status", "ok");
    let mut gone_at = None;
    let t0 = Instant::now();
    while t0.elapsed() < DEADLINE + Duration::from_secs(10) {
        if gone_at.is_none() && old.iter().all(|p| !alive(*p)) {
            gone_at = Some(std::time::SystemTime::now());
        }
        if recorded(&fx, "describe").len() > describes_before {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let out = resumed.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let gone_at = gone_at.expect("old processes were gone before the resumed coordinator's first call");
    let first_new = recorded(&fx, "describe").into_iter().map(|pid| {
        let t: f64 = std::fs::read_to_string(fx.pids.join(format!("describe-{pid}"))).unwrap().parse().unwrap();
        t
    }).fold(0.0f64, f64::max);
    let gone = gone_at.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    assert!(first_new + 0.05 >= gone, "a new adapter call started {:.3} s before the old group was gone", gone - first_new);
    // open cleanup: a supervisor (stand-in for an earlier run) holds the lease; the gate refuses
    let mut holder = Command::new(IMOPT).args(["supervise", "--lease", fx.dir.join("st/measure.lease").to_str().unwrap(),
        "--", "sleep", "30"]).stdin(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let blocked = Command::new(IMOPT).args(["status", "--state", fx.dir.join("st").to_str().unwrap(), "--goal",
        fx.dir.join("goal.json").to_str().unwrap(), "--", "python3", fx.script.to_str().unwrap()])
        .env("FIXTURE_DIR", &fx.pids).env("IMOPT_GATE_WAIT_S", "1").output().unwrap();
    assert!(!blocked.status.success());
    assert!(String::from_utf8_lossy(&blocked.stderr).contains("no new measurement starts"));
    drop(holder.stdin.take()); // its lifeline closes: it ends `sleep` and releases the lease
    let _ = holder.wait();
    let ok = Command::new(IMOPT).args(["status", "--state", fx.dir.join("st").to_str().unwrap(), "--goal",
        fx.dir.join("goal.json").to_str().unwrap(), "--", "python3", fx.script.to_str().unwrap()])
        .env("FIXTURE_DIR", &fx.pids).env("IMOPT_GATE_WAIT_S", "1").output().unwrap();
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stderr));
}

/// An interrupted measurement is journaled once when the next coordinator opens the state, charged to the
/// budget, and never counted as a measurement; no completed step runs twice.
#[test]
fn an_interrupted_test_is_journaled_and_charged_and_nothing_completed_runs_twice() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("interrupted");
    let mut c = imopt(&fx, "cycle", "stubborn");
    wait_for("the adapter's first run", Duration::from_secs(30), || !recorded(&fx, "run-child").is_empty());
    std::thread::sleep(Duration::from_millis(500));
    sigkill(&mut c);
    let st = imopt(&fx, "status", "ok").wait_with_output().unwrap();
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let again = imopt(&fx, "status", "ok").wait_with_output().unwrap();
    assert!(again.status.success());
    let journal: Vec<ironmule_optimizer::json::Value> = std::fs::read_to_string(fx.dir.join("st/journal.jsonl")).unwrap()
        .lines().filter_map(|l| ironmule_optimizer::json::parse(l).ok()).collect();
    let kinds = |k: &str| journal.iter().filter(|r| r.str("kind") == Some(k)).count();
    let interrupted: Vec<_> = journal.iter().filter(|r| r.str("kind") == Some("attempt_interrupted")).collect();
    assert_eq!(interrupted.len(), 1, "recorded once, not again by the second open");
    let charged = interrupted[0].get("body").unwrap().num("seconds").unwrap();
    assert!(charged >= 0.5, "charged {charged} s");
    assert_eq!(interrupted[0].get("body").unwrap().str("what"), Some("reference"));
    assert_eq!(kinds("reference"), 0, "the interrupted run is no measurement");
}

/// SIGTERM (also SIGINT, SIGHUP) to a supervisor ends its adapter group first; a supervisor whose
/// lifeline is already closed when it holds the lease starts nothing.
#[test]
fn a_signalled_supervisor_cleans_up_and_a_closed_lifeline_starts_nothing() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("signal");
    let lease = fx.dir.join("measure.lease");
    let mut sup = Command::new(IMOPT).args(["supervise", "--lease", lease.to_str().unwrap(), "--", "python3",
        fx.script.to_str().unwrap(), "run"]).env("FIXTURE_DIR", &fx.pids).env("FIXTURE_MODE", "hang")
        .stdin(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap();
    wait_for("adapter and descendant", Duration::from_secs(30), || !recorded(&fx, "run-child").is_empty());
    let own: Vec<u32> = recorded(&fx, "run").into_iter().chain(recorded(&fx, "run-child")).collect();
    Command::new("kill").args(["-TERM", &sup.id().to_string()]).status().unwrap();
    let t0 = Instant::now();
    while sup.try_wait().unwrap().is_none() {
        assert!(t0.elapsed() < DEADLINE, "the supervisor did not finish");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(own.iter().all(|p| !alive(*p)), "its group was not ended before it exited");
    drop(sup.stdin.take());
    // lifeline closed from the start (stdin at EOF): no adapter process at all
    let fx = fixture("closed");
    let out = Command::new(IMOPT).args(["supervise", "--lease", fx.dir.join("measure.lease").to_str().unwrap(), "--",
        "python3", fx.script.to_str().unwrap(), "run"]).env("FIXTURE_DIR", &fx.pids).env("FIXTURE_MODE", "hang")
        .stdin(Stdio::null()).output().unwrap();
    assert_eq!(out.status.code(), Some(125));
    std::thread::sleep(Duration::from_millis(300));
    assert!(recorded(&fx, "run").is_empty(), "an adapter started after its coordinator was gone");
}

/// The graceful path: a coordinator killed while an adapter that handles SIGTERM is measuring; the
/// supervisor's SIGTERM reaches it (its handler leaves a mark) and it ends before the grace runs out, so
/// SIGKILL was not needed. Without the SIGTERM step this test fails (no mark).
#[test]
fn a_cooperative_adapter_ends_on_sigterm_within_the_grace() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture("graceful");
    let mut c = imopt(&fx, "cycle", "graceful");
    wait_for("the adapter's first run", Duration::from_secs(30), || !recorded(&fx, "run").is_empty());
    std::thread::sleep(Duration::from_millis(300)); // its handler is installed before it sleeps
    let own = recorded(&fx, "run");
    sigkill(&mut c);
    let took = wait_for("the adapter gone", DEADLINE, || own.iter().all(|p| !alive(*p)));
    assert_eq!(recorded(&fx, "term"), own, "SIGTERM reached the adapter and its handler ran");
    assert!(took < GRACE, "ended by SIGTERM within the grace, not by the later SIGKILL ({took:?})");
}
