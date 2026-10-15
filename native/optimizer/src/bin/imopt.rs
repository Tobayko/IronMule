//! `imopt`: the OPT1 coordinator's local entry point (opt-in; nothing runs unless called).
//!
//! ```text
//! imopt detect
//! imopt cycle    --state DIR --goal GOAL.json -- ADAPTER [ARGS...]   full cycle for one application adapter
//! imopt status   --state DIR --goal GOAL.json -- ADAPTER [ARGS...]
//! imopt rollback --state DIR --goal GOAL.json -- ADAPTER [ARGS...]
//! imopt compare  --state DIR --worker IMOPT_CPUJOB [--budget S] [--holdout N] [--chunks N] [--out FILE]
//!                equal-budget comparison of the sequential decision (backlog OPT1), then activation
//!                of a policy that passed, or the reference kept
//! imopt dream-replay --state DIR                     re-run the recorded dream choice from the journal
//! imopt managed  --state DIR --worker IMOPT_CPUJOB --seed N   one background job under the active
//!                concurrency policy (reference: one thread) beside the foreground probe
//! imopt managed-rollback --state DIR
//! imopt campaign --state DIR --worker IMOPT_CPUJOB --baseline-threads T [--rounds 2] [--budget 300]
//!                [--holdout 10] [--seed 1] [--no-dream] [--reward sparse|dense] [--start cold|warm]
//!                [--holdout-seed0 N] [--status]
//!                resumable learning campaign (backlog OPT3): train -> holdout against the fixed
//!                baseline -> decision -> replay selection -> next round; restart continues it
//! imopt duel --state DIR --worker IMOPT_CPUJOB --base CAMPAIGN_DIR --candidate CAMPAIGN_DIR --seeds A-B
//!                both campaigns' final learners on the same schedules (OPT2 phase D: with against
//!                without dream); resumable, an interrupted schedule is burned
//! imopt rulecheck --state DIR --worker IMOPT_CPUJOB --goal goals/cpujob.json [--cycles 30] [--seed N]
//!                [--control] [--status]   protocol 1 against protocol 2 on the same triplets (OPT4b)
//! imopt calibrate --state DIR --worker IMOPT_CPUJOB [--reps 3] [--slices 10] [--threads 1,2,3,4,6,8]
//!                headroom of the background-thread decision (backlog OPT3 step B): oracle per load
//!                level against the best fixed thread count
//! ```

use ironmule_optimizer::adapter::Adapter;
use ironmule_optimizer::calibrate;
use ironmule_optimizer::coordinator::{Coordinator, Outcome};
use ironmule_optimizer::dream::{self, ReplayModel};
use ironmule_optimizer::episode::{self, EpisodeConfig, Policy, Transition, Workers};
use ironmule_optimizer::experiment::{self, Frozen, Setup};
use ironmule_optimizer::identity;
use ironmule_optimizer::json::{self, Value};
use ironmule_optimizer::rl::QTable;
use ironmule_optimizer::sha256;
use ironmule_optimizer::store::{self, EnvelopeError, Journal};
use std::path::{Path, PathBuf};
use std::time::Duration;

use ironmule_optimizer::campaign::{self, Campaign, Plan, CONCURRENCY_SCHEMA};

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let end = args.iter().position(|a| a == "--").unwrap_or(args.len());
    args[..end].iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn need<'a>(args: &'a [String], name: &str) -> &'a str {
    opt(args, name).unwrap_or_else(|| fail(&format!("missing {name}")))
}

fn fail(msg: &str) -> ! {
    eprintln!("imopt: {msg}");
    std::process::exit(2)
}

fn adapter_argv(args: &[String]) -> Vec<String> {
    match args.iter().position(|a| a == "--") {
        Some(i) if i + 1 < args.len() => args[i + 1..].to_vec(),
        _ => fail("give the adapter command after --"),
    }
}

fn coordinator(args: &[String]) -> Coordinator {
    let timeout = opt(args, "--timeout").and_then(|t| t.parse().ok()).unwrap_or(3600.0);
    let state = Path::new(need(args, "--state"));
    std::fs::create_dir_all(state).unwrap_or_else(|e| fail(&e.to_string()));
    // Every adapter call supervised under this state's lease; nothing starts while an earlier run's
    // measurement processes are not gone (backlog OPT3-D (c)).
    let adapter = Adapter::new(adapter_argv(args), Duration::from_secs_f64(timeout))
        .supervised(std::env::current_exe().unwrap_or_else(|e| fail(&e.to_string())), state.join("measure.lease"))
        .unwrap_or_else(|e| fail(&e));
    Coordinator::open(Path::new(need(args, "--state")), adapter, Path::new(need(args, "--goal")), None)
        .unwrap_or_else(|e| fail(&e))
}

/// The concurrency setup fixed in backlog OPT1 (job size from `--chunks`, default 3000 x 256 KiB;
/// thread set from `--threads`, default OPT1's 1-4), always bounded by the logical CPUs.
fn episode_config(args: &[String]) -> EpisodeConfig {
    episode_config_with(args, &[1, 2, 3, 4])
}

fn episode_config_with(args: &[String], default_threads: &[usize]) -> EpisodeConfig {
    let cpus = identity::logical_cpus().unwrap_or(1);
    let threads: Vec<usize> = match opt(args, "--threads") {
        Some(t) => t.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| fail("bad --threads"))).collect(),
        None => default_threads.to_vec(),
    };
    EpisodeConfig {
        job_chunks: opt(args, "--chunks").and_then(|c| c.parse().ok()).unwrap_or(3000),
        chunk_size: 256 * 1024,
        slice_ms: 500,
        max_slices: 60,
        rates: [0.0, 20.0, 60.0],
        fg_size: 256 * 1024,
        penalty: 4.0,
        threads: threads.into_iter().filter(|&t| t >= 1 && t <= cpus.max(1)).collect(),
    }
}


/// OPT4b: protocol 1 and protocol 2 judged on the same triplets (backlog OPT4b); resumable, never reruns.
fn rulecheck_cmd(args: &[String]) {
    use ironmule_optimizer::coordinator::Goal;
    use ironmule_optimizer::rulecheck;
    let dir = PathBuf::from(need(args, "--state"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let worker = need(args, "--worker").to_string();
    let goal = Goal::load(Path::new(need(args, "--goal"))).unwrap_or_else(|e| fail(&e));
    let control = args.iter().any(|a| a == "--control");
    let cycles: usize = opt(args, "--cycles").and_then(|x| x.parse().ok()).unwrap_or(30);
    let seed: u64 = opt(args, "--seed").and_then(|x| x.parse().ok()).unwrap_or(20261013);
    let cpus = identity::logical_cpus().unwrap_or(1);
    let large_threads = [1, 2, 3, 4, 6, 8].into_iter().filter(|&t| t <= cpus).max().unwrap_or(1);
    // The 1-thread arms and the hog fill the fastest core class: on Apple silicon the performance cores
    // (a full machine would push reference runs onto efficiency cores at random, a noise source of its own).
    let hog = identity::performance_cpus().unwrap_or(cpus).min(cpus).saturating_sub(1).max(1);
    // The study's own worker environment is part of its plan; anything that changes runs is refused.
    if std::env::var_os("IMOPT_CPUJOB_PAD").is_some() {
        fail("IMOPT_CPUJOB_PAD is set: it would pad every arm; the study pads only the sub arm's references itself");
    }
    if let Ok(f) = std::env::var("IMOPT_FAULT") {
        if f.split(',').any(|x| !x.starts_with("rulecheck_abort:")) {
            fail("IMOPT_FAULT is set: injected faults would reach the measured runs");
        }
    }
    let cells = rulecheck::cells(control);
    let plan = rulecheck::schedule(&cells, cycles, seed);
    let need_large = (9 * cycles).div_ceil(10);
    let (host, _) = identity::detect(None);
    let worker_digest = std::fs::read(&worker).map(|b| sha256::hex_digest(&b)).unwrap_or_default();
    let plan_json = Value::obj().with("cells", cells.iter().map(|(e, c)| format!("{e}/{c}")).collect::<Vec<_>>())
        .with("cycles_per_cell", cycles).with("seed", format!("{seed}")).with("large_threads", large_threads).with("hog_threads", hog)
        .with("sub_pad", rulecheck::SUB_PAD).with("need_large", need_large).with("goal", goal.raw.clone())
        .with("chunks", std::env::var("IMOPT_CPUJOB_CHUNKS").unwrap_or_default())
        .with("chunk_kb", std::env::var("IMOPT_CPUJOB_CHUNK_KB").unwrap_or_default());
    let id = sha256::hex_digest(Value::obj().with("host", host.digest.as_str()).with("worker", worker_digest.as_str())
        .with("plan", plan_json.clone()).dump().as_bytes());
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let stored = journal.of_kind("rulecheck_plan").find(|r| r.str("identity") == Some(id.as_str())).and_then(|r| r.get("body")).cloned();
    let reference_digest = match stored {
        Some(b) => b.str("reference_digest").unwrap_or("").to_string(),
        None => {
            if !journal.records.is_empty() {
                fail("this state holds another study (other plan, host or worker); a fresh run needs a new --state");
            }
            if args.iter().any(|a| a == "--status") {
                fail("no study in this state");
            }
            let reference = Adapter::new(vec![worker.clone()], Duration::from_secs(600))
                .supervised(std::env::current_exe().unwrap_or_else(|e| fail(&e.to_string())), dir.join("measure.lease"))
                .unwrap_or_else(|e| fail(&e));
            let r = ironmule_optimizer::adapter::run(&reference, "threads=1", 1);
            if !r.ok {
                fail("the reference run failed");
            }
            journal.append(&id, "rulecheck_plan", plan_json.clone().with("host", host.facts.clone())
                .with("reference_digest", r.correctness.as_str()).with("load_average", identity::load_average())).unwrap_or_else(|e| fail(&e));
            r.correctness
        }
    };
    let mut log = |m: &str| eprintln!("imopt: {m}");
    let mut setup = rulecheck::Setup { journal: &mut journal, id, worker, rule: goal.rule, large_threads, hog, seed,
                                       plan, reference_digest, log: &mut log, lease: Some(dir.join("measure.lease")) };
    if !args.iter().any(|a| a == "--status") {
        setup.run().unwrap_or_else(|e| fail(&e));
    }
    let analysis = setup.analysis(&cells, cycles, need_large).with("plan", plan_json).with("load_average_now", identity::load_average());
    std::fs::write(dir.join("rulecheck-report.json"), analysis.dump()).unwrap_or_else(|e| fail(&e.to_string()));
    println!("{}", analysis.dump());
}

/// OPT3: a resumable learning campaign (one per state directory; its plan is fixed at the first start).
fn campaign_cmd(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let worker = need(args, "--worker").to_string();
    let cfg = episode_config(args);
    let id = concurrency_identity(&cfg, &worker);
    let num = |name: &str, d: f64| opt(args, name).map_or(d, |x| x.parse().unwrap_or_else(|_| fail(&format!("bad {name}"))));
    let baseline_threads = num("--baseline-threads", f64::NAN) as usize;
    let baseline = match cfg.threads.iter().position(|&t| t == baseline_threads) {
        Some(b) => b,
        None if args.iter().any(|a| a == "--status") => 0, // the stored plan supplies it
        None => fail(&format!("--baseline-threads must be one of {:?}", cfg.threads)),
    };
    let plan = Plan { rounds: num("--rounds", 2.0) as usize, budget_s: num("--budget", 300.0), holdout: num("--holdout", 10.0) as usize,
                      baseline, seed: num("--seed", 1.0) as u64, dream: !args.iter().any(|a| a == "--no-dream"),
                      dense: match opt(args, "--reward") { None | Some("sparse") => false, Some("dense") => true,
                                                           Some(_) => fail("--reward is sparse or dense") },
                      warm: match opt(args, "--start") { None | Some("cold") => false, Some("warm") => true,
                                                         Some(_) => fail("--start is cold or warm") },
                      holdout_seed0: opt(args, "--holdout-seed0").map(|x| x.parse().unwrap_or_else(|_| fail("bad --holdout-seed0"))) };
    if plan.holdout_seed0.is_some() && plan.rounds > 1 && plan.holdout > 0 {
        fail("--holdout-seed0 fixes one round's holdout seeds; use it with --rounds 1 or --holdout 0");
    }
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let stored = journal.of_kind("campaign_plan").find(|r| r.str("identity") == Some(id.as_str())).and_then(|r| r.get("body")).cloned();
    let mut workers = None;
    let status_only = args.iter().any(|a| a == "--status");
    let plan = match (&stored, status_only) {
        (Some(b), true) => b.get("plan").and_then(Plan::from_json).unwrap_or(plan),
        _ => plan,
    };
    let (targets, reference_digest) = match stored {
        Some(b) => {
            if b.get("plan") != Some(&plan.to_json()) {
                fail("this state holds a campaign with another plan; a fresh campaign needs a new --state");
            }
            let t: Vec<f64> = b.arr("foreground_targets_ms").map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::INFINITY)).collect())
                .unwrap_or_default();
            ([t[0], t[1], t[2]], b.str("digest").unwrap_or("").to_string())
        }
        None => {
            if !journal.records.is_empty() {
                fail("this state belongs to another identity or run (changed host, worker or setup): its experience and \
                      qualification do not transfer; a fresh campaign needs a new --state");
            }
            if args.iter().any(|a| a == "--status") {
                fail("no campaign in this state");
            }
            let mut w = Workers::start(&worker).unwrap_or_else(|e| fail(&e));
            let (ref_s, digest) = w.job_alone(&cfg, 1).unwrap_or_else(|e| fail(&e));
            // OPT2 phase D: a campaign compared with another must share its foreground targets
            let source = opt(args, "--targets-from");
            let targets = match source {
                Some(src) => {
                    let (t, d) = stored_targets(Path::new(src), &id);
                    if d != digest {
                        fail("--targets-from: that campaign holds another reference output");
                    }
                    t
                }
                None => episode::calibrate(&mut w, &cfg, 6).unwrap_or_else(|e| fail(&e)),
            };
            let (host, _) = identity::detect(None);
            journal.append(&id, "campaign_plan", Value::obj().with("plan", plan.to_json()).with("episode_config", cfg.to_json())
                .with("targets_copied", source.is_some())
                .with("host", host.facts).with("reference_job_s", ref_s).with("digest", digest.as_str())
                .with("foreground_targets_ms", targets.to_vec()).with("load_average", identity::load_average())).unwrap_or_else(|e| fail(&e));
            workers = Some(w);
            (targets, digest)
        }
    };
    let mut log = |m: &str| eprintln!("imopt: {m}");
    let mut c = Campaign { journal: &mut journal, id: id.clone(), cfg: cfg.clone(), plan, targets, reference_digest, log: &mut log };
    if args.iter().any(|a| a == "--status") {
        println!("{}", c.status().dump());
        return;
    }
    let mut w = workers.unwrap_or_else(|| Workers::start(&worker).unwrap_or_else(|e| fail(&e)));
    let status = c.run(&mut w, &dir).unwrap_or_else(|e| fail(&e));
    println!("{}", status.dump());
}

/// A campaign's stored foreground targets and reference digest (identity must match).
fn stored_targets(dir: &Path, id: &str) -> ([f64; 3], String) {
    let journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let body = journal.of_kind("campaign_plan").find(|r| r.str("identity") == Some(id)).and_then(|r| r.get("body")).cloned()
        .unwrap_or_else(|| fail(&format!("{}: no campaign of this identity (host, worker and setup must match)", dir.display())));
    let t: Vec<f64> = body.arr("foreground_targets_ms").map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::INFINITY)).collect())
        .unwrap_or_default();
    if t.len() != 3 {
        fail("bad stored targets");
    }
    ([t[0], t[1], t[2]], body.str("digest").unwrap_or("").to_string())
}

/// A finished campaign's final learner (the one its last round judged), its targets and reference digest.
fn final_learner(dir: &Path, id: &str, cfg: &EpisodeConfig) -> (ironmule_optimizer::rl::QTable, [f64; 3], String) {
    use ironmule_optimizer::campaign::{learned_digest, Campaign, Plan};
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let body = journal.of_kind("campaign_plan").find(|r| r.str("identity") == Some(id)).and_then(|r| r.get("body")).cloned()
        .unwrap_or_else(|| fail(&format!("{}: no campaign of this identity (host, worker and setup must match)", dir.display())));
    if journal.of_kind("campaign_done").find(|r| r.str("identity") == Some(id)).is_none() {
        fail(&format!("{}: the campaign has not finished", dir.display()));
    }
    let plan = body.get("plan").and_then(Plan::from_json).unwrap_or_else(|| fail("bad stored plan"));
    let t: Vec<f64> = body.arr("foreground_targets_ms").map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::INFINITY)).collect())
        .unwrap_or_default();
    let targets = [t[0], t[1], t[2]];
    let digest = body.str("digest").unwrap_or("").to_string();
    let last = journal.of_kind("round_decision").filter(|r| r.str("identity") == Some(id)).last()
        .and_then(|r| r.get("body")).and_then(|b| b.str("learned_sha256")).map(String::from);
    let mut log = |_: &str| {};
    let c = Campaign { journal: &mut journal, id: id.to_string(), cfg: cfg.clone(), plan: plan.clone(), targets,
                       reference_digest: digest.clone(), log: &mut log };
    let table = c.rebuild(c.config_for(plan.rounds));
    if last.as_deref() != Some(learned_digest(&table).as_str()) {
        fail(&format!("{}: the rebuilt learner differs from the one its last round judged", dir.display()));
    }
    (table, targets, digest)
}

/// OPT2 phase D: a campaign's final learner against another's on the same schedules (backlog OPT2).
fn duel_cmd(args: &[String]) {
    use ironmule_optimizer::campaign::Duel;
    let dir = PathBuf::from(need(args, "--state"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let worker = need(args, "--worker").to_string();
    let cfg = episode_config(args);
    let id = concurrency_identity(&cfg, &worker);
    let (base, targets, digest) = final_learner(Path::new(need(args, "--base")), &id, &cfg);
    let (candidate, cand_targets, cand_digest) = final_learner(Path::new(need(args, "--candidate")), &id, &cfg);
    if digest != cand_digest {
        fail("the two campaigns hold different reference outputs");
    }
    if Value::from(targets.to_vec()).dump() != Value::from(cand_targets.to_vec()).dump() {
        fail("the two campaigns trained against different foreground targets (start the candidate with --targets-from)");
    }
    let seeds: Vec<u64> = match need(args, "--seeds").split_once('-').map(|(a, b)| (a.parse::<u64>(), b.parse::<u64>())) {
        Some((Ok(a), Ok(b))) if a <= b => (a..=b).collect(),
        _ => fail("--seeds is A-B"),
    };
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let mut log = |m: &str| eprintln!("imopt: {m}");
    let mut w = Workers::start(&worker).unwrap_or_else(|e| fail(&e));
    let mut d = Duel { journal: &mut journal, id, cfg, targets, reference_digest: digest, base, candidate, seeds, log: &mut log };
    println!("{}", d.run(&mut w).unwrap_or_else(|e| fail(&e)).dump());
}

/// OPT3 step B: the headroom calibration, journaled slice by slice; the analysis is the report.
fn calibrate_cmd(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let worker = need(args, "--worker").to_string();
    let cfg = episode_config_with(args, &[1, 2, 3, 4, 6, 8]);
    let reps: usize = opt(args, "--reps").and_then(|x| x.parse().ok()).unwrap_or(3);
    let slices: usize = opt(args, "--slices").and_then(|x| x.parse().ok()).unwrap_or(10);
    let id = concurrency_identity(&cfg, &worker);
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let (host, _) = identity::detect(None);
    let load_before = identity::load_average();
    let mut w = Workers::start(&worker).unwrap_or_else(|e| fail(&e));
    let (ref_s, reference_digest) = w.job_alone(&cfg, 1).unwrap_or_else(|e| fail(&e));
    let targets = episode::calibrate(&mut w, &cfg, 6).unwrap_or_else(|e| fail(&e));
    journal.append(&id, "calibration_start", Value::obj().with("host", host.facts.clone()).with("episode_config", cfg.to_json())
        .with("reps", reps).with("slices_per_block", slices).with("reference_job_s", ref_s).with("digest", reference_digest.as_str())
        .with("foreground_targets_ms", targets.to_vec()).with("load_average", load_before.clone())).unwrap_or_else(|e| fail(&e));
    eprintln!("imopt: reference job {ref_s:.2} s; targets {targets:?} ms; {} blocks of {slices} slices",
              calibrate::plan(cfg.threads.len(), reps).len());
    let rows = calibrate::run(&mut w, &cfg, &targets, &reference_digest, reps, slices,
                              |r| journal.append(&id, "calibration_slice", r.to_json()).map(|_| ())).unwrap_or_else(|e| fail(&e));
    let analysis = calibrate::analyse(&rows, &cfg, 2000, 20261013).with("load_average_before", load_before)
        .with("load_average_after", identity::load_average()).with("host", host.facts).with("hardware_class", host.hardware_class.as_str())
        .with("episode_config", cfg.to_json()).with("foreground_targets_ms", targets.to_vec()).with("reference_job_s", ref_s);
    journal.append(&id, "calibration_analysis", analysis.clone()).unwrap_or_else(|e| fail(&e));
    let out = opt(args, "--out").map(PathBuf::from).unwrap_or_else(|| dir.join("calibration-report.json"));
    std::fs::write(&out, analysis.dump()).unwrap_or_else(|e| fail(&e.to_string()));
    println!("{}", analysis.dump());
}

fn concurrency_identity(cfg: &EpisodeConfig, worker: &str) -> String {
    let (host, _) = identity::detect(None);
    let worker_digest = std::fs::read(worker).map(|b| sha256::hex_digest(&b)).unwrap_or_default();
    sha256::hex_digest(Value::obj().with("host", host.digest).with("worker", worker_digest).with("config", cfg.to_json())
        .dump().as_bytes())
}

fn compare(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let worker = need(args, "--worker").to_string();
    let cfg = episode_config(args);
    let id = concurrency_identity(&cfg, &worker);
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    if journal.of_kind("campaign_plan").next().is_some() {
        fail("this state holds a learning campaign; the fresh comparison needs its own --state");
    }
    let mut log = |m: &str| eprintln!("imopt: {m}");
    let mut w = Workers::start(&worker).unwrap_or_else(|e| fail(&e));
    let (host, caps) = identity::detect(None);
    journal.append(&id, "detect", Value::obj().with("host", host.facts.clone()).with("hardware_class", host.hardware_class.as_str())
        .with("capabilities", caps.iter().map(|c| c.to_json()).collect::<Vec<_>>()).with("episode_config", cfg.to_json()))
        .unwrap_or_else(|e| fail(&e));
    // Reference: the job alone with one thread gives the output every episode must reproduce.
    let (ref_s, reference_digest) = w.job_alone(&cfg, 1).unwrap_or_else(|e| fail(&e));
    let targets = episode::calibrate(&mut w, &cfg, 6).unwrap_or_else(|e| fail(&e));
    journal.append(&id, "reference", Value::obj().with("job_alone_1_thread_s", ref_s).with("digest", reference_digest.as_str())
        .with("foreground_targets_ms", targets.to_vec())).unwrap_or_else(|e| fail(&e));
    log(&format!("reference job {ref_s:.2} s; foreground targets {targets:?} ms"));
    let budget_s: f64 = opt(args, "--budget").and_then(|b| b.parse().ok()).unwrap_or(600.0);
    let holdout_n: u64 = opt(args, "--holdout").and_then(|b| b.parse().ok()).unwrap_or(20);
    let mut setup = Setup { cfg: cfg.clone(), targets, reference_digest, budget_s, holdout: (1..=holdout_n).collect(),
                            train_seed0: 1000, identity: id.clone(), journal: &mut journal, log: &mut log };
    let mut trained = Vec::new();
    for strategy in ["sweep", "planner", "qlearn", "qlearn_dream"] {
        (setup.log)(&format!("training {strategy} for {budget_s} s"));
        let t = match strategy {
            "sweep" => experiment::sweep(&mut setup, &mut w),
            "planner" => experiment::planner(&mut setup, &mut w),
            "qlearn" => experiment::qlearn(&mut setup, &mut w),
            _ => experiment::qlearn_dream(&mut setup, &mut w),
        }.unwrap_or_else(|e| fail(&e));
        if let Frozen::Q(table) = &t.policy {
            store::write_envelope(&dir.join(format!("{}-qtable.json", t.name)), "imopt.qtable.v1", &id, table.to_json())
                .unwrap_or_else(|e| fail(&e));
        }
        setup.journal.append(&id, "trained", Value::obj().with("strategy", t.name.as_str()).with("episodes", t.episodes)
            .with("spent_s", t.spent_s).with("notes", t.notes.clone()).with("policy", t.policy.describe(&cfg)))
            .unwrap_or_else(|e| fail(&e));
        trained.push(t);
    }
    let mut policies: Vec<(String, Frozen)> = trained.iter().map(|t| (t.name.clone(), t.policy.clone())).collect();
    policies.insert(1, ("load_rule".into(), Frozen::LoadRule));
    let h = experiment::holdout(&mut setup, &mut w, policies).unwrap_or_else(|e| fail(&e));
    let analysis = experiment::analyse(&h);
    setup.journal.append(&id, "holdout_analysis", analysis.clone()).unwrap_or_else(|e| fail(&e));
    // Activation: a learned policy that passed its fixed rule against the best simple strategy, else reference.
    let rl = analysis.get("rl_vs_best_simple").and_then(|d| d.str("verdict")) == Some("accepted");
    let dr = analysis.get("dream_vs_best_simple").and_then(|d| d.str("verdict")) == Some("accepted")
        && analysis.get("dream_vs_rl").and_then(|d| d.str("verdict")) == Some("accepted");
    let chosen = if dr { Some("qlearn_dream") } else if rl { Some("qlearn") } else { None };
    let activation = match chosen {
        Some(name) => {
            let t = trained.iter().find(|t| t.name == name).unwrap();
            activate_concurrency(&dir, &id, t.policy.describe(&cfg).with("strategy", name));
            Value::obj().with("result", "activated").with("strategy", name)
        }
        None => Value::obj().with("result", "reference_kept")
            .with("reason", "no learned policy passed the fixed holdout rule against the best simple strategy"),
    };
    setup.journal.append(&id, "outcome", activation.clone()).unwrap_or_else(|e| fail(&e));
    let report = Value::obj().with("identity", id.as_str()).with("hardware_class", host.hardware_class.as_str())
        .with("host", host.facts).with("episode_config", cfg.to_json()).with("foreground_targets_ms", targets.to_vec())
        .with("reference_job_s", ref_s).with("budget_s", budget_s)
        .with("training", trained.iter().map(|t| Value::obj().with("strategy", t.name.as_str()).with("episodes", t.episodes)
            .with("spent_s", t.spent_s).with("notes", t.notes.clone())).collect::<Vec<_>>())
        .with("holdout", h).with("analysis", analysis).with("activation", activation);
    let out = opt(args, "--out").map(PathBuf::from).unwrap_or_else(|| dir.join("compare-report.json"));
    std::fs::write(&out, report.dump()).unwrap_or_else(|e| fail(&e.to_string()));
    println!("{}", report.get("analysis").unwrap().dump());
}

fn activate_concurrency(dir: &Path, id: &str, policy: Value) {
    campaign::activate_concurrency(dir, id, policy).unwrap_or_else(|e| fail(&e));
}

fn frozen_from(policy: &Value) -> Option<Frozen> {
    match policy.str("kind")? {
        "load_rule" => Some(Frozen::LoadRule),
        "q_greedy" => QTable::from_json(policy.get("table")?).map(Frozen::Q),
        _ => None,
    }
}

fn managed(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    let worker = need(args, "--worker");
    let cfg = episode_config(args);
    let id = concurrency_identity(&cfg, worker);
    let path = dir.join("concurrency-active.json");
    let (mut frozen, source) = match store::read_envelope(&path, CONCURRENCY_SCHEMA, &id) {
        Ok(p) => match p.get("policy").and_then(frozen_from) {
            Some(f) => (f, "active policy"),
            None => (Frozen::Fixed(0), "reference (active policy unreadable)"),
        },
        Err(EnvelopeError::Missing) => (Frozen::Fixed(0), "reference (nothing active)"),
        Err(e) => {
            store::quarantine(&path);
            eprintln!("imopt: active concurrency policy not valid here ({e:?}); quarantined, reference used");
            (Frozen::Fixed(0), "reference (foreign or damaged policy quarantined)")
        }
    };
    let mut w = Workers::start(worker).unwrap_or_else(|e| fail(&e));
    let targets = episode::calibrate(&mut w, &cfg, 3).unwrap_or_else(|e| fail(&e));
    let seed: u64 = need(args, "--seed").parse().unwrap_or_else(|_| fail("bad --seed"));
    let mut policy = match &mut frozen {
        Frozen::Fixed(a) => Policy::Fixed(*a),
        Frozen::LoadRule => Policy::LoadRule,
        Frozen::Q(t) => Policy::Q { table: t, explore: false },
    };
    let r = episode::run_episode(&mut w, &cfg, &targets, seed, &mut policy).unwrap_or_else(|e| fail(&e));
    println!("{}", r.to_json(false).with("policy_source", source).dump());
}

fn managed_rollback(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    let path = dir.join("concurrency-active.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| fail("nothing active"));
    let v = json::parse(text.trim()).unwrap_or_else(|e| fail(&e));
    let id = v.str("identity").unwrap_or("").to_string();
    let previous = store::read_envelope(&path, CONCURRENCY_SCHEMA, &id).ok()
        .and_then(|p| p.get("previous").cloned()).unwrap_or(Value::Null);
    let moved = store::quarantine(&path);
    if previous != Value::Null {
        store::write_envelope(&path, CONCURRENCY_SCHEMA, &id, Value::obj().with("policy", previous).with("previous", Value::Null))
            .unwrap_or_else(|e| fail(&e));
    }
    let mut journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    journal.append(&id, "rollback", Value::obj().with("what", "concurrency policy")
        .with("restored_previous", path.exists()).with("moved_to", moved.map(|p| p.display().to_string())))
        .unwrap_or_else(|e| fail(&e));
    println!("{}", Value::obj().with("ok", true).with("restored_previous", path.exists()).dump());
}

/// Reproduce the dream choice from the journal's recorded trees: same records, same answer.
fn dream_replay(args: &[String]) {
    let dir = PathBuf::from(need(args, "--state"));
    let journal = Journal::open(&dir.join("journal.jsonl")).unwrap_or_else(|e| fail(&e));
    let mut episodes: std::collections::BTreeMap<u64, Vec<Transition>> = Default::default();
    let cut = journal.of_kind("dream_select").next().and_then(|r| r.num("seq")).unwrap_or(f64::INFINITY);
    for r in journal.of_kind("tree_node") {
        let b = r.get("body").unwrap();
        if b.str("strategy") != Some("qlearn_dream") || b.str("phase") != Some("train") || r.num("seq").unwrap_or(0.0) > cut {
            continue;
        }
        if let (Some(seed), Some(t)) = (b.num("episode_seed"), Transition::from_json(b)) {
            episodes.entry(seed as u64).or_default().push(t);
        }
    }
    let trees: Vec<Vec<Transition>> = episodes.into_values().collect();
    let actions = trees.iter().flatten().map(|t| t.action + 1).max().unwrap_or(1)
        .max(journal.of_kind("detect").next().and_then(|r| r.get("body")).and_then(|b| b.get("episode_config"))
            .and_then(|c| c.arr("threads")).map_or(1, |t| t.len()));
    let dev = ReplayModel::build(trees.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, t)| &t[..]));
    let val = ReplayModel::build(trees.iter().enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, t)| &t[..]));
    let (chosen, report) = dream::select(&dev, &val, actions, 60, 20, &[1, 2, 3]);
    let recorded = journal.of_kind("dream_select").next().and_then(|r| r.get("body")).and_then(|b| b.get("chosen")).cloned();
    let chosen = chosen.map(|c| c.to_json()).unwrap_or(Value::Null);
    println!("{}", Value::obj().with("trees", trees.len()).with("chosen", chosen.clone())
        .with("matches_recorded_choice", recorded.as_ref() == Some(&chosen)).with("report", report).dump());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("detect") => {
            let (id, caps) = identity::detect(None);
            println!("{}", Value::obj().with("identity", id.digest.as_str()).with("hardware_class", id.hardware_class.as_str())
                .with("facts", id.facts).with("capabilities", caps.iter().map(|c| c.to_json()).collect::<Vec<_>>()).dump());
        }
        Some("cycle") => {
            let mut c = coordinator(&args);
            let outcome = c.cycle().unwrap_or_else(|e| fail(&e));
            let v = match outcome {
                Outcome::Activated { action, mean_ratio } => Value::obj().with("result", "activated").with("action", action).with("mean_ratio", mean_ratio),
                Outcome::ReferenceKept { reason } => Value::obj().with("result", "reference_kept").with("reason", reason),
            };
            println!("{}", v.dump());
        }
        Some("status") => println!("{}", coordinator(&args).status().dump()),
        Some("rollback") => {
            let mut c = coordinator(&args);
            let done = c.rollback("requested").unwrap_or_else(|e| fail(&e));
            println!("{}", Value::obj().with("rolled_back", done).dump());
        }
        Some("compare") => compare(&args),
        Some("managed") => managed(&args),
        Some("managed-rollback") => managed_rollback(&args),
        Some("dream-replay") => dream_replay(&args),
        Some("calibrate") => calibrate_cmd(&args),
        Some("campaign") => campaign_cmd(&args),
        Some("duel") => duel_cmd(&args),
        Some("rulecheck") => rulecheck_cmd(&args),
        #[cfg(unix)]
        Some("supervise") => {
            // Internal: one supervised adapter call (see `proc`). `imopt supervise --lease FILE -- ARGV...`
            let lease = PathBuf::from(need(&args, "--lease"));
            std::process::exit(ironmule_optimizer::proc::supervise(&lease, &adapter_argv(&args)));
        }
        _ => fail("usage: imopt detect|cycle|status|rollback|compare|managed|managed-rollback|dream-replay|calibrate|campaign|rulecheck"),
    }
}
