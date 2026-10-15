//! Adapter 2 (OPT1): an own CPU job whose reversible application setting is its thread count.
//!
//! Adapter protocol (see `adapter.rs`): `describe`, `run --action threads=N --seed S`, `apply --action`,
//! `revert --previous JSON`, `active`. The setting lives in a JSON config file
//! (`IMOPT_CPUJOB_CONFIG`, default `./imopt-cpujob.json`) that the job reads when started without an
//! explicit thread count; no file means the job's default of one thread.
//! Long-lived workers for episodes: `bg` (background job in time slices: stdin `new SEED CHUNKS SIZE`,
//! `slice THREADS MS`, `digest`) and `fg` (foreground probe: stdin `probe RATE MS SIZE SEED`, reply `e2e_ms`, `service_ms`,
//! and since PROBE1 `lateness_ms` and `prep_ms` per request, with `e2e` = lateness + prep + service).
//! Test-only fault injection: `IMOPT_FAULT=crash:ACTION | hang:ACTION | wrong:ACTION`.
//! Study-only (OPT4): `IMOPT_CPUJOB_PAD=F` makes `run` sleep F x its job time after the job and count
//! it in the cost (a synthetic, output-identical slowdown); `hog THREADS [PARENT_PID]` runs the job forever
//! as injected background load and exits once its parent is gone.

use ironmule_optimizer::cpujob::{self, Job, JobSpec};
use ironmule_optimizer::identity;
use ironmule_optimizer::json::{self, Value};
use ironmule_optimizer::sha256;
use std::io::{BufRead, Write};
use std::time::Duration;

fn spec() -> JobSpec {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    JobSpec { seed: 20261013, chunks: env("IMOPT_CPUJOB_CHUNKS", 256), chunk_size: env("IMOPT_CPUJOB_CHUNK_KB", 256) * 1024 }
}

fn config_path() -> std::path::PathBuf {
    std::env::var("IMOPT_CPUJOB_CONFIG").map(Into::into).unwrap_or_else(|_| "imopt-cpujob.json".into())
}

fn thread_choices() -> Vec<usize> {
    let cpus = identity::logical_cpus().unwrap_or(1);
    [1, 2, 3, 4, 6, 8, 12, 16].into_iter().filter(|&t| t <= cpus.max(1)).collect()
}

fn parse_action(a: &str) -> Option<usize> {
    let t: usize = a.strip_prefix("threads=")?.parse().ok()?;
    thread_choices().contains(&t).then_some(t)
}

fn fault(kind: &str, action: &str) -> bool {
    std::env::var("IMOPT_FAULT").ok().is_some_and(|f| f.split(',').any(|x| x == format!("{kind}:{action}")))
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn read_config() -> Option<Value> {
    std::fs::read_to_string(config_path()).ok().and_then(|t| json::parse(t.trim()).ok())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = |v: Value| println!("{}", v.dump());
    match args.first().map(String::as_str) {
        Some("describe") => {
            let s = spec();
            let me = std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).map(|b| sha256::hex_digest(&b));
            let actions: Vec<Value> = thread_choices().into_iter().map(|t| Value::obj().with("id", format!("threads={t}"))
                .with("settings", Value::obj().with("threads", t)).with("class", "auto")).collect();
            out(Value::obj().with("name", "imopt-cpujob").with("reference", "threads=1").with("actions", actions)
                .with("metric", "wall_s").with("evaluator", "coordinator")
                .with("identity", Value::obj().with("app", "imopt-cpujob").with("version", env!("CARGO_PKG_VERSION"))
                    .with("binary_sha256", me).with("seed", s.seed).with("chunks", s.chunks).with("chunk_size", s.chunk_size)));
        }
        Some("run") => {
            let action = arg(&args, "--action").unwrap_or("");
            let Some(threads) = parse_action(action) else {
                return out(Value::obj().with("ok", false).with("error", format!("unknown action {action}")));
            };
            if fault("crash", action) {
                std::process::exit(3);
            }
            if fault("hang", action) {
                std::thread::sleep(Duration::from_secs(3600));
            }
            let r = cpujob::run_job(spec(), threads);
            let pad: f64 = std::env::var("IMOPT_CPUJOB_PAD").ok().and_then(|p| p.parse().ok()).filter(|p: &f64| *p > 0.0).unwrap_or(0.0);
            std::thread::sleep(Duration::from_secs_f64(pad * r.wall_s));
            let cost = r.wall_s * (1.0 + pad);
            let digest = if fault("wrong", action) { "0".repeat(64) } else { r.digest };
            out(Value::obj().with("ok", r.round_trip_ok).with("cost", cost).with("correctness", digest)
                .with("metrics", Value::obj().with("threads", threads).with("bytes_in", r.bytes_in).with("bytes_out", r.bytes_out)
                    .with("mb_per_s", r.bytes_in as f64 / r.wall_s / 1e6).with("pad", pad))
                .with("error", (!r.round_trip_ok).then_some("round trip failed")));
        }
        Some("apply") => {
            let action = arg(&args, "--action").unwrap_or("");
            let Some(threads) = parse_action(action) else {
                return out(Value::obj().with("ok", false).with("error", "unknown action"));
            };
            let previous = read_config().map_or(Value::Null, |c| Value::obj().with("config", c.clone())
                .with("action", c.num("threads").map(|t| format!("threads={t}"))));
            let ok = ironmule_optimizer::store::write_atomic(&config_path(),
                format!("{}\n", Value::obj().with("threads", threads).dump()).as_bytes()).is_ok();
            out(Value::obj().with("ok", ok).with("previous", previous));
        }
        Some("revert") => {
            let previous = arg(&args, "--previous").and_then(|p| json::parse(p).ok()).unwrap_or(Value::Null);
            let ok = match previous.get("config") {
                Some(c) => ironmule_optimizer::store::write_atomic(&config_path(), format!("{}\n", c.dump()).as_bytes()).is_ok(),
                None => match std::fs::remove_file(config_path()) {
                    Ok(()) => true,
                    Err(e) => e.kind() == std::io::ErrorKind::NotFound,
                },
            };
            out(Value::obj().with("ok", ok));
        }
        Some("active") => {
            let action = read_config().and_then(|c| c.num("threads")).map(|t| format!("threads={t}"));
            out(Value::obj().with("ok", true).with("action", action));
        }
        Some("job") => {
            // The application itself: runs the job with its configured (or default) setting.
            let threads = arg(&args, "--threads").and_then(|t| t.parse().ok())
                .or_else(|| read_config().and_then(|c| c.num("threads")).map(|t| t as usize)).unwrap_or(1);
            let r = cpujob::run_job(spec(), threads);
            out(Value::obj().with("threads", threads).with("wall_s", r.wall_s).with("digest", r.digest));
        }
        Some("hog") => {
            let threads = args.get(1).and_then(|t| t.parse().ok()).unwrap_or(1);
            // Never outlive the harness: a killed parent must not leave injected load behind. The parent
            // passes its pid (reading it here could already see the reparented pid after a fast exit).
            #[cfg(unix)]
            let parent: u32 = args.get(2).and_then(|p| p.parse().ok()).unwrap_or_else(std::os::unix::process::parent_id);
            loop {
                #[cfg(unix)]
                if std::os::unix::process::parent_id() != parent {
                    std::process::exit(0);
                }
                std::hint::black_box(cpujob::run_job(spec(), threads));
            }
        }
        Some("bg") => {
            let mut job: Option<Job> = None;
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                let w: Vec<&str> = line.split_whitespace().collect();
                let reply = match w.as_slice() {
                    ["new", seed, chunks, size] => {
                        job = Some(Job::new(JobSpec { seed: seed.parse().unwrap_or(1), chunks: chunks.parse().unwrap_or(1),
                                                      chunk_size: size.parse().unwrap_or(65536) }));
                        Value::obj().with("ok", true)
                    }
                    ["slice", threads, ms] => match &job {
                        Some(j) => {
                            let t0 = std::time::Instant::now();
                            let done = j.run_for(threads.parse().unwrap_or(1), Some(Duration::from_millis(ms.parse().unwrap_or(500))));
                            Value::obj().with("ok", true).with("chunks", done).with("done", j.done()).with("finished", j.finished())
                                .with("slice_s", t0.elapsed().as_secs_f64())
                        }
                        None => Value::obj().with("ok", false).with("error", "no job"),
                    },
                    ["digest"] => match job.as_ref().and_then(|j| j.digest().map(|d| (d, j.round_trip(0)))) {
                        Some(((d, n), rt)) => Value::obj().with("ok", true).with("digest", d).with("bytes_out", n).with("round_trip", rt),
                        None => Value::obj().with("ok", false).with("error", "job not finished"),
                    },
                    _ => Value::obj().with("ok", false).with("error", "unknown command"),
                };
                println!("{}", reply.dump());
                let _ = std::io::stdout().flush();
            }
        }
        Some("fg") => {
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                let w: Vec<&str> = line.split_whitespace().collect();
                let reply = match w.as_slice() {
                    ["probe", rate, ms, size, seed] => {
                        let lat = cpujob::probe(seed.parse().unwrap_or(1), rate.parse().unwrap_or(0.0),
                                                Duration::from_millis(ms.parse().unwrap_or(500)), size.parse().unwrap_or(65536));
                        Value::obj().with("ok", true).with("e2e_ms", lat.e2e_ms).with("service_ms", lat.service_ms)
                            .with("lateness_ms", lat.lateness_ms).with("prep_ms", lat.prep_ms)
                    }
                    _ => Value::obj().with("ok", false).with("error", "unknown command"),
                };
                println!("{}", reply.dump());
                let _ = std::io::stdout().flush();
            }
        }
        _ => {
            eprintln!("usage: imopt-cpujob describe|run|apply|revert|active|job|hog|bg|fg");
            std::process::exit(2);
        }
    }
}
