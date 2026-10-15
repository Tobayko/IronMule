//! Real episodes for the sequential decision (OPT1): a background compression job runs in slices
//! beside a foreground probe on the same CPUs. Both are long-lived `imopt-cpujob` workers (`bg`, `fg`);
//! every slice is one real transition. Nothing here is simulated.

use crate::cpujob::percentile;
use crate::json::{self, Value};
use crate::rl::{QTable, State};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

/// The background job's corpus seed: the same job in every episode, so its output digest is fixed.
pub const JOB_SEED: u64 = 20261013;

#[derive(Clone, Debug)]
pub struct EpisodeConfig {
    pub job_chunks: usize,
    pub chunk_size: usize,
    pub slice_ms: u64,
    pub max_slices: usize,
    /// Foreground request rate per level (idle, light, heavy), requests per second.
    pub rates: [f64; 3],
    pub fg_size: usize,
    pub penalty: f64,
    /// Thread counts the action indices stand for (bounded by the detected logical CPUs).
    pub threads: Vec<usize>,
}

impl EpisodeConfig {
    pub fn to_json(&self) -> Value {
        Value::obj().with("job_chunks", self.job_chunks).with("chunk_size", self.chunk_size).with("slice_ms", self.slice_ms)
            .with("max_slices", self.max_slices).with("rates", self.rates.to_vec()).with("fg_size", self.fg_size)
            .with("penalty", self.penalty).with("threads", self.threads.clone())
            .with("probe", "v2: violations on the p95 from scheduled arrival to completion")
    }
}

/// The seeded foreground schedule: levels per slice, a Markov chain that stays with probability 0.7.
pub fn schedule(seed: u64, slices: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut level = 0u8;
    (0..slices).map(|_| {
        let u = next();
        if u >= 0.7 {
            level = (level + if u < 0.85 { 1 } else { 2 }) % 3;
        }
        level
    }).collect()
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    fn spawn(binary: &str, mode: &str) -> Result<Worker, String> {
        let mut child = Command::new(binary).arg(mode).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
            .spawn().map_err(|e| format!("cannot start {binary} {mode}: {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Ok(Worker { child, stdin, stdout })
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.stdin, "{line}").and_then(|_| self.stdin.flush()).map_err(|e| format!("worker write: {e}"))
    }

    fn read(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| format!("worker read: {e}"))?;
        if line.is_empty() {
            return Err("worker exited".into());
        }
        let v = json::parse(line.trim())?;
        if v.boolean("ok") != Some(true) {
            return Err(format!("worker error: {}", v.dump()));
        }
        Ok(v)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct Workers {
    bg: Worker,
    fg: Worker,
}

impl Workers {
    pub fn start(binary: &str) -> Result<Workers, String> {
        Ok(Workers { bg: Worker::spawn(binary, "bg")?, fg: Worker::spawn(binary, "fg")? })
    }

    /// Foreground alone (background idle), one slice: end-to-end latencies in ms.
    pub fn probe_alone(&mut self, cfg: &EpisodeConfig, level: usize, seed: u64) -> Result<Vec<f64>, String> {
        self.fg.send(&format!("probe {} {} {} {seed}", cfg.rates[level], cfg.slice_ms, cfg.fg_size))?;
        latencies(&self.fg.read()?, "e2e_ms")
    }

    /// Start a fresh background job (the fixed corpus).
    pub fn new_job(&mut self, cfg: &EpisodeConfig) -> Result<(), String> {
        self.bg.send(&format!("new {JOB_SEED} {} {}", cfg.job_chunks, cfg.chunk_size))?;
        self.bg.read().map(|_| ())
    }

    /// One real slice: the background job with `threads` beside the foreground probe at `level`.
    pub fn slice(&mut self, cfg: &EpisodeConfig, level: u8, threads: usize, probe_seed: u64) -> Result<Slice, String> {
        self.bg.send(&format!("slice {threads} {}", cfg.slice_ms))?;
        self.fg.send(&format!("probe {} {} {} {probe_seed}", cfg.rates[level as usize], cfg.slice_ms, cfg.fg_size))?;
        let bg = self.bg.read()?;
        let fg = self.fg.read()?;
        let (e2e, service) = (latencies(&fg, "e2e_ms")?, latencies(&fg, "service_ms")?);
        Ok(Slice { chunks: bg.num("chunks").unwrap_or(0.0) as usize, done: bg.num("done").unwrap_or(0.0) as usize,
                   finished: bg.boolean("finished") == Some(true), slice_s: bg.num("slice_s").unwrap_or(0.0),
                   requests: e2e.len(), e2e_p95: percentile(&e2e, 0.95), service_p95: percentile(&service, 0.95) })
    }

    /// Finish the current job with every thread outside any measurement, then its output digest.
    pub fn finish_job(&mut self, cfg: &EpisodeConfig) -> Result<String, String> {
        loop {
            self.bg.send(&format!("slice {} 1000", cfg.threads[cfg.threads.len() - 1]))?;
            if self.bg.read()?.boolean("finished") == Some(true) {
                break;
            }
        }
        self.bg.send("digest")?;
        Ok(self.bg.read()?.str("digest").unwrap_or("").to_string())
    }

    /// The job alone with a fixed thread count, to its end: (seconds, digest).
    pub fn job_alone(&mut self, cfg: &EpisodeConfig, threads: usize) -> Result<(f64, String), String> {
        self.bg.send(&format!("new {JOB_SEED} {} {}", cfg.job_chunks, cfg.chunk_size))?;
        self.bg.read()?;
        let t0 = Instant::now();
        loop {
            self.bg.send(&format!("slice {threads} {}", cfg.slice_ms))?;
            if self.bg.read()?.boolean("finished") == Some(true) {
                break;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        self.bg.send("digest")?;
        let d = self.bg.read()?;
        Ok((secs, d.str("digest").unwrap_or("").to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct Slice {
    pub chunks: usize,
    pub done: usize,
    pub finished: bool,
    pub slice_s: f64,
    pub requests: usize,
    pub e2e_p95: Option<f64>,
    pub service_p95: Option<f64>,
}

fn latencies(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    v.arr(key).ok_or(format!("no {key}"))?.iter().map(|x| x.as_f64().ok_or("bad latency".to_string())).collect()
}

/// Foreground target per level: 1.5 x the p95 with the background idle (fixed in OPT1).
pub fn calibrate(workers: &mut Workers, cfg: &EpisodeConfig, slices_per_level: usize) -> Result<[f64; 3], String> {
    let mut targets = [f64::INFINITY; 3];
    for level in 1..3 {
        let mut all = Vec::new();
        for k in 0..slices_per_level {
            all.extend(workers.probe_alone(cfg, level, 77 + k as u64)?);
        }
        targets[level] = 1.5 * percentile(&all, 0.95).ok_or("no foreground samples")?;
    }
    Ok(targets)
}

pub enum Policy<'a> {
    Fixed(usize),
    /// idle -> all threads, light -> half, heavy -> one; unknown (first slice) -> half.
    LoadRule,
    Q { table: &'a mut QTable, explore: bool },
}

impl Policy<'_> {
    pub fn name(&self) -> String {
        match self {
            Policy::Fixed(a) => format!("fixed[{a}]"),
            Policy::LoadRule => "load_rule".into(),
            Policy::Q { explore, .. } => format!("q{}", if *explore { "_explore" } else { "_greedy" }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transition {
    pub t: usize,
    pub state: State,
    pub action: usize,
    pub reward: f64,
    pub next: Option<State>,
    pub level: u8,
    /// Foreground p95, scheduled arrival to completion (probe v2); the violation is judged on it.
    pub p95_ms: Option<f64>,
    /// Foreground p95 of the processing time alone.
    pub service_p95_ms: Option<f64>,
    pub violated: bool,
    pub chunks: usize,
    pub slice_s: f64,
}

impl Transition {
    pub fn to_json(&self) -> Value {
        Value::obj().with("t", self.t).with("state", self.state.to_json()).with("action", self.action)
            .with("reward", self.reward).with("next", self.next.map(State::to_json)).with("level", self.level as u64)
            .with("p95_ms", self.p95_ms).with("service_p95_ms", self.service_p95_ms).with("violated", self.violated).with("chunks", self.chunks).with("slice_s", self.slice_s)
    }

    pub fn from_json(v: &Value) -> Option<Transition> {
        Some(Transition {
            t: v.num("t")? as usize,
            state: State::from_json(v.get("state")?)?,
            action: v.num("action")? as usize,
            reward: v.num("reward")?,
            next: match v.get("next")? { Value::Null => None, n => Some(State::from_json(n)?) },
            level: v.num("level")? as u8,
            p95_ms: v.num("p95_ms"),
            service_p95_ms: v.num("service_p95_ms"),
            violated: v.boolean("violated")?,
            chunks: v.num("chunks")? as usize,
            slice_s: v.num("slice_s")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EpisodeResult {
    pub seed: u64,
    pub policy: String,
    pub ret: f64,
    pub slices: usize,
    pub violations: usize,
    pub finished: bool,
    pub digest: String,
    pub wall_s: f64,
    pub transitions: Vec<Transition>,
}

impl EpisodeResult {
    pub fn to_json(&self, with_transitions: bool) -> Value {
        let v = Value::obj().with("seed", self.seed).with("policy", self.policy.as_str()).with("return", self.ret)
            .with("slices", self.slices).with("violations", self.violations).with("finished", self.finished)
            .with("digest", self.digest.as_str()).with("wall_s", self.wall_s);
        if with_transitions { v.with("transitions", self.transitions.iter().map(Transition::to_json).collect::<Vec<_>>()) } else { v }
    }
}

fn progress(done: usize, total: usize) -> u8 {
    ((4 * done) / total.max(1)).min(3) as u8
}

/// OPT2's dense training reward: `-1 - penalty x min(2, 4 x max(0, p95 / target - 1))`; no foreground
/// request (idle) or an infinite target counts as no excess.
pub fn dense_reward(p95: Option<f64>, target: f64, penalty: f64) -> f64 {
    let excess = p95.filter(|_| target.is_finite() && target > 0.0).map_or(0.0, |p| (p / target - 1.0).max(0.0));
    -1.0 - penalty * (4.0 * excess).min(2.0)
}

/// One real episode. The Q policy learns online when `explore` is set (one update per slice).
pub fn run_episode(workers: &mut Workers, cfg: &EpisodeConfig, targets: &[f64; 3], seed: u64,
                   policy: &mut Policy) -> Result<EpisodeResult, String> {
    let levels = schedule(seed, cfg.max_slices);
    let all: Vec<usize> = (0..cfg.threads.len()).collect();
    workers.new_job(cfg)?;
    let t0 = Instant::now();
    let (mut state, mut ret, mut violations) = (State { level: 3, progress: 0, violated: false }, 0.0, 0usize);
    let mut transitions = Vec::new();
    let mut finished = false;
    for (t, &level) in levels.iter().enumerate() {
        let a = match policy {
            Policy::Fixed(a) => *a,
            Policy::LoadRule => {
                let n = cfg.threads.len();
                match state.level { 0 => n - 1, 2 => 0, _ => (n - 1) / 2 }
            }
            Policy::Q { table, explore } => table.choose(state, *explore, &all),
        };
        let sl = workers.slice(cfg, level, cfg.threads[a], seed * 1000 + t as u64)?;
        let violated = sl.e2e_p95.is_some_and(|p| p > targets[level as usize]);
        finished = sl.finished;
        let reward = -1.0 - cfg.penalty * violated as u8 as f64;
        ret += reward; // the return (product measure) is always the sparse one
        // a learner may train on the dense reward instead (OPT2); its transitions carry what it learned from
        let reward = match policy {
            Policy::Q { table, explore: true } if table.dense => dense_reward(sl.e2e_p95, targets[level as usize], cfg.penalty),
            _ => reward,
        };
        violations += violated as usize;
        let next = (!finished && t + 1 < levels.len()).then(|| State { level, progress: progress(sl.done, cfg.job_chunks), violated });
        if let Policy::Q { table, explore: true } = policy {
            table.update(state, a, reward, next);
        }
        transitions.push(Transition { t, state, action: a, reward, next, level, p95_ms: sl.e2e_p95, service_p95_ms: sl.service_p95,
                                      violated, chunks: sl.chunks, slice_s: sl.slice_s });
        match next {
            Some(n) => state = n,
            None => break,
        }
    }
    if let Policy::Q { table, explore: true } = policy {
        table.end_episode();
    }
    let wall_s = t0.elapsed().as_secs_f64();
    // Finish an unfinished job outside the episode so its digest can still be checked.
    let digest = workers.finish_job(cfg)?;
    Ok(EpisodeResult { seed, policy: policy.name(), ret, slices: transitions.len(), violations, finished,
                       digest, wall_s, transitions })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_reward_grows_with_the_excess_and_is_capped() {
        assert_eq!(dense_reward(None, 5.0, 4.0), -1.0);
        assert_eq!(dense_reward(Some(4.0), 5.0, 4.0), -1.0);
        assert_eq!(dense_reward(Some(9.0), f64::INFINITY, 4.0), -1.0);
        assert!((dense_reward(Some(5.5), 5.0, 4.0) - (-1.0 - 4.0 * 0.4)).abs() < 1e-12);
        assert_eq!(dense_reward(Some(50.0), 5.0, 4.0), -9.0, "capped at twice the penalty");
    }

    #[test]
    fn schedules_are_seeded_and_use_every_level() {
        assert_eq!(schedule(5, 50), schedule(5, 50));
        assert_ne!(schedule(5, 50), schedule(6, 50));
        let s = schedule(9, 400);
        for level in 0..3 {
            let share = s.iter().filter(|&&x| x == level).count() as f64 / 400.0;
            assert!(share > 0.15, "level {level} share {share}");
        }
    }
}
