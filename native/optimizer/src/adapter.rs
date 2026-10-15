//! Adapter protocol. An adapter is an executable (argv prefix) the coordinator calls once per step,
//! each call a fresh child process, so a measurement cannot inherit another's state and a crash ends
//! one call, not the coordinator. The adapter prints one JSON object as its last stdout line.
//!
//! ```text
//! describe                      -> {name, identity:{...}, reference, actions:[{id, settings, class}], metric,
//!                                   evaluator: "coordinator" | "adapter"}
//!                                   class: "auto" (may qualify) | "tested_only" (numerics change: measured,
//!                                   never auto-qualified)
//! run --action ID --seed N      -> {ok, cost, correctness, metrics, error?}   (cost: lower is better)
//! qualify --action ID           -> {ok, qualified, evidence}                  (evaluator "adapter" only)
//! apply --action ID             -> {ok, previous}                             (previous: opaque JSON)
//! revert --previous JSON        -> {ok}
//! active                        -> {ok, action}                               (null = reference/default)
//! ```

use crate::json::{self, Value};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Adapter {
    pub argv: Vec<String>,
    pub timeout: Duration,
    /// Run each call under `imopt supervise` holding this lease (see `proc`); None runs it directly.
    pub supervision: Option<Supervision>,
}

#[derive(Clone, Debug)]
pub struct Supervision {
    pub exe: std::path::PathBuf,
    pub lease: std::path::PathBuf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub id: String,
    pub settings: Value,
    pub auto: bool,
}

#[derive(Clone, Debug)]
pub struct Description {
    pub name: String,
    pub identity: Value,
    pub reference: String,
    pub actions: Vec<Action>,
    pub metric: String,
    pub adapter_evaluator: bool,
    pub raw: Value,
}

#[derive(Debug, PartialEq)]
pub enum CallError {
    Spawn(String),
    Timeout,
    Crashed(String),
    BadOutput(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Spawn(e) => write!(f, "adapter could not start: {e}"),
            CallError::Timeout => write!(f, "adapter timed out"),
            CallError::Crashed(e) => write!(f, "adapter crashed: {e}"),
            CallError::BadOutput(e) => write!(f, "adapter output unusable: {e}"),
        }
    }
}

/// Every child the coordinator or a study harness spawns goes through this lock, so a child started
/// by another thread cannot inherit a pipe end created in the same moment (a pipe whose close-on-exec
/// flag is set only after its creation). A child holding an adapter's stdout would block that call.
pub static SPAWN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl Adapter {
    pub fn new(argv: Vec<String>, timeout: Duration) -> Adapter {
        Adapter { argv, timeout, supervision: None }
    }

    /// Supervise every call through this executable (`imopt`) under the state's lease, after waiting for
    /// the processes of an earlier run of the same state to be gone (refused after the gate wait;
    /// `IMOPT_GATE_WAIT_S` overrides it for tests). Unix only; elsewhere calls stay unsupervised.
    pub fn supervised(mut self, exe: std::path::PathBuf, lease: std::path::PathBuf) -> Result<Adapter, String> {
        if cfg!(unix) {
            let wait = std::env::var("IMOPT_GATE_WAIT_S").ok().and_then(|s| s.parse().ok())
                .map(Duration::from_secs_f64).unwrap_or(crate::proc::GATE_WAIT);
            crate::proc::await_quiescence(&lease, wait)?;
            self.supervision = Some(Supervision { exe, lease });
        }
        Ok(self)
    }

    /// One call in a fresh child. stderr is inherited for diagnostics; stdout's last line is the answer.
    pub fn call(&self, args: &[&str]) -> Result<Value, CallError> {
        let (program, rest) = self.argv.split_first().ok_or(CallError::Spawn("empty adapter command".into()))?;
        let spawned = {
            let _lock = SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            match &self.supervision {
                // The supervisor's stdin is the lifeline: we hold its only write end until the call ends.
                Some(s) => {
                    let mut cmd = Command::new(&s.exe);
                    cmd.arg("supervise").arg("--lease").arg(&s.lease).arg("--").args(&self.argv).args(args)
                        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
                    #[cfg(unix)]
                    std::os::unix::process::CommandExt::process_group(&mut cmd, 0); // terminal signals reach us, not it
                    cmd.spawn()
                }
                None => Command::new(program).args(rest).args(args).stdin(Stdio::null()).stdout(Stdio::piped())
                    .stderr(Stdio::inherit()).spawn(),
            }
        };
        let mut child = spawned.map_err(|e| CallError::Spawn(e.to_string()))?;
        let lifeline = child.stdin.take();
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stdout.read_to_string(&mut text);
            text
        });
        let start = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if start.elapsed() > self.timeout => {
                    if lifeline.is_some() {
                        // Closing the lifeline makes the supervisor end the adapter's group; wait until it has
                        // (it exits only then), so no later call starts beside a leftover. Never kill it: its
                        // death would release the lease before the group is proven gone.
                        drop(lifeline);
                        let waited = Instant::now();
                        let mut told = false;
                        while child.try_wait().ok().flatten().is_none() {
                            if !told && waited.elapsed() > crate::proc::DEADLINE {
                                eprintln!("imopt: a timed-out adapter's processes are not gone yet; waiting");
                                told = true;
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    } else {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    return Err(CallError::Timeout);
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                Err(e) => return Err(CallError::Crashed(e.to_string())),
            }
        };
        let text = reader.join().unwrap_or_default();
        if !status.success() {
            return Err(CallError::Crashed(format!("exit {status}")));
        }
        let line = text.lines().rev().find(|l| !l.trim().is_empty()).ok_or(CallError::BadOutput("no output".into()))?;
        let v = json::parse(line.trim()).map_err(CallError::BadOutput)?;
        if !matches!(v, Value::Obj(_)) {
            return Err(CallError::BadOutput("not an object".into()));
        }
        Ok(v)
    }

    pub fn describe(&self) -> Result<Description, CallError> {
        let v = self.call(&["describe"])?;
        let bad = |w: &str| CallError::BadOutput(format!("describe: {w}"));
        let reference = v.str("reference").ok_or(bad("no reference"))?.to_string();
        let mut actions = Vec::new();
        for a in v.arr("actions").ok_or(bad("no actions"))? {
            let id = a.str("id").ok_or(bad("action without id"))?.to_string();
            let class = a.str("class").unwrap_or("auto");
            if class != "auto" && class != "tested_only" {
                return Err(bad("unknown action class"));
            }
            if actions.iter().any(|x: &Action| x.id == id) {
                return Err(bad("duplicate action id"));
            }
            actions.push(Action { id, settings: a.get("settings").cloned().unwrap_or(Value::Null), auto: class == "auto" });
        }
        if !actions.iter().any(|a| a.id == reference) {
            return Err(bad("reference is not an action"));
        }
        if actions.len() > 64 {
            return Err(bad("more than 64 actions: the action set must stay bounded"));
        }
        Ok(Description {
            name: v.str("name").ok_or(bad("no name"))?.to_string(),
            identity: v.get("identity").cloned().unwrap_or(Value::Null),
            reference,
            actions,
            metric: v.str("metric").unwrap_or("cost").to_string(),
            adapter_evaluator: v.str("evaluator") == Some("adapter"),
            raw: v,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub ok: bool,
    pub cost: f64,
    pub correctness: String,
    pub metrics: Value,
    pub error: Option<String>,
    pub seconds: f64,
}

impl RunResult {
    pub fn to_json(&self) -> Value {
        Value::obj().with("ok", self.ok).with("cost", self.cost).with("correctness", self.correctness.as_str())
            .with("metrics", self.metrics.clone()).with("error", self.error.clone()).with("seconds", self.seconds)
    }
}

/// Run one action; any failure (crash, timeout, bad output, ok=false, non-positive cost) is a
/// failed result with its reason, never a measurement.
pub fn run(adapter: &Adapter, action: &str, seed: u64) -> RunResult {
    let start = Instant::now();
    let seed = seed.to_string();
    let answer = adapter.call(&["run", "--action", action, "--seed", &seed]);
    let seconds = start.elapsed().as_secs_f64();
    let fail = |e: String| RunResult { ok: false, cost: f64::NAN, correctness: String::new(), metrics: Value::Null, error: Some(e), seconds };
    match answer {
        Err(e) => fail(e.to_string()),
        Ok(v) => {
            let cost = v.num("cost").unwrap_or(f64::NAN);
            if v.boolean("ok") != Some(true) {
                return fail(v.str("error").unwrap_or("adapter reported failure").to_string());
            }
            if !(cost.is_finite() && cost > 0.0) {
                return fail("non-positive or missing cost".into());
            }
            RunResult { ok: true, cost, correctness: v.str("correctness").unwrap_or("").to_string(),
                        metrics: v.get("metrics").cloned().unwrap_or(Value::Null), error: None, seconds }
        }
    }
}
