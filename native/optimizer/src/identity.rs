//! Capabilities and exact identity of the machine and the coordinator.
//!
//! Each platform has its own probe (`cfg(target_os)`); nothing is assumed across them. A capability
//! is one of six explicit states: `unknown` (not probed or the probe failed), `unsupported` (this
//! milestone may not act on it, e.g. kernel or power settings, which need a separately authorised
//! adapter), `observable` (read, not changed), `controllable` (a bounded action exists), `tested`
//! (a real run used it) and `qualified` (an independent check passed on this exact identity).
//! The identity digest is the SHA-256 of the canonical JSON of the stable facts plus the coordinator
//! binary's own digest, so a changed machine, OS build or coordinator invalidates learned state.

use crate::json::Value;
use crate::sha256;
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapState {
    Unknown,
    Unsupported,
    Observable,
    Controllable,
    Tested,
    Qualified,
}

impl CapState {
    pub fn name(self) -> &'static str {
        match self {
            CapState::Unknown => "unknown",
            CapState::Unsupported => "unsupported",
            CapState::Observable => "observable",
            CapState::Controllable => "controllable",
            CapState::Tested => "tested",
            CapState::Qualified => "qualified",
        }
    }

    pub fn parse(s: &str) -> Option<CapState> {
        Some(match s {
            "unknown" => CapState::Unknown,
            "unsupported" => CapState::Unsupported,
            "observable" => CapState::Observable,
            "controllable" => CapState::Controllable,
            "tested" => CapState::Tested,
            "qualified" => CapState::Qualified,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Capability {
    pub name: String,
    pub state: CapState,
    pub value: Value,
    pub note: String,
}

impl Capability {
    fn new(name: &str, state: CapState, value: Value, note: &str) -> Capability {
        Capability { name: name.into(), state, value, note: note.into() }
    }

    pub fn to_json(&self) -> Value {
        Value::obj().with("name", self.name.as_str()).with("state", self.state.name())
            .with("value", self.value.clone()).with("note", self.note.as_str())
    }
}

/// Facts that define identity (stable across reboots of the same machine and software).
#[derive(Clone, Debug)]
pub struct Identity {
    pub facts: Value,
    pub digest: String,
    /// Coarse class for ordering transfer only (os, arch, cpu model, logical cpus): never for qualification.
    pub hardware_class: String,
}

#[cfg_attr(target_os = "windows", allow(dead_code))]
fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// The host's 1/5/15-minute load averages, recorded beside measurements (ambient load is not controlled).
pub fn load_average() -> Value {
    let text = std::fs::read_to_string("/proc/loadavg").ok().or_else(|| {
        std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).replace(['{', '}'], ""))
    });
    let v: Vec<f64> = text.unwrap_or_default().split_whitespace().take(3).filter_map(|x| x.replace(',', ".").parse().ok()).collect();
    if v.len() == 3 { Value::from(v) } else { Value::Null }
}

/// Logical CPUs this process may use (respects affinity and, on Linux, cgroup quotas in std).
pub fn logical_cpus() -> Option<usize> {
    std::thread::available_parallelism().ok().map(|n| n.get())
}

/// The fastest core class's logical CPUs where the OS reports core classes (Apple silicon:
/// `hw.perflevel0.logicalcpu`, the performance cores); None elsewhere (symmetric or unknown).
pub fn performance_cpus() -> Option<usize> {
    if cfg!(target_os = "macos") {
        run("sysctl", &["-n", "hw.perflevel0.logicalcpu"]).and_then(|s| s.parse().ok())
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn platform_facts() -> Vec<(String, Option<Value>)> {
    let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
    let model = cpuinfo.lines().find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string());
    let mem = read("/proc/meminfo").and_then(|m| m.lines().find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|k| k.parse::<u64>().ok()).map(|k| k * 1024));
    let os = read("/etc/os-release").and_then(|r| r.lines().find(|l| l.starts_with("PRETTY_NAME="))
        .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string()));
    let kernel = read("/proc/sys/kernel/osrelease").map(|s| s.trim().to_string());
    let quota = read("/sys/fs/cgroup/cpu.max").map(|s| s.trim().to_string());
    let gpu = run("nvidia-smi", &["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]);
    vec![
        ("cpu_model".into(), model.map(Value::from)),
        ("memory_bytes".into(), mem.map(Value::from)),
        ("os".into(), os.map(Value::from)),
        ("kernel".into(), kernel.map(Value::from)),
        ("cgroup_cpu_max".into(), quota.map(Value::from)),
        ("gpu".into(), gpu.map(Value::from)),
    ]
}

#[cfg(target_os = "macos")]
fn platform_facts() -> Vec<(String, Option<Value>)> {
    let sysctl = |k: &str| run("sysctl", &["-n", k]);
    vec![
        ("cpu_model".into(), sysctl("machdep.cpu.brand_string").map(Value::from)),
        ("memory_bytes".into(), sysctl("hw.memsize").and_then(|s| s.parse::<u64>().ok()).map(Value::from)),
        ("os".into(), run("sw_vers", &["-productVersion"]).map(|v| Value::from(format!("macOS {v}")))),
        ("kernel".into(), sysctl("kern.osrelease").map(Value::from)),
        ("gpu".into(), (std::env::consts::ARCH == "aarch64").then(|| Value::from("Apple GPU (unified memory)"))),
    ]
}

#[cfg(target_os = "windows")]
fn platform_facts() -> Vec<(String, Option<Value>)> {
    // Windows probe of this milestone: environment only. Memory size and GPU need Win32 calls that
    // were not implemented or tested; they stay unknown rather than guessed.
    vec![
        ("cpu_model".into(), std::env::var("PROCESSOR_IDENTIFIER").ok().map(Value::from)),
        ("memory_bytes".into(), None),
        ("os".into(), std::env::var("OS").ok().map(Value::from)),
        ("kernel".into(), None),
        ("gpu".into(), None),
    ]
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_facts() -> Vec<(String, Option<Value>)> {
    vec![]
}

/// The coordinator binary's own digest (code identity); unknown if it cannot be read.
fn self_digest() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    std::fs::read(exe).ok().map(|b| sha256::hex_digest(&b))
}

/// Probe the machine. `code_digest` overrides the binary digest (tests, or an embedding caller).
pub fn detect(code_digest: Option<&str>) -> (Identity, Vec<Capability>) {
    let mut facts = Value::obj()
        .with("platform", std::env::consts::OS)
        .with("arch", std::env::consts::ARCH)
        .with("logical_cpus", logical_cpus())
        .with("coordinator_version", env!("CARGO_PKG_VERSION"))
        // A build with test-only fault injection is another identity: its states never mix with a product build's.
        .with("test_faults_build", cfg!(feature = "test-faults"));
    let mut caps = vec![
        Capability::new("platform", CapState::Observable, Value::from(std::env::consts::OS), "compile-time target"),
        match logical_cpus() {
            Some(n) => Capability::new("cpu.logical", CapState::Observable, Value::from(n), "available_parallelism"),
            None => Capability::new("cpu.logical", CapState::Unknown, Value::Null, "available_parallelism failed"),
        },
    ];
    for (key, value) in platform_facts() {
        match value {
            Some(v) => {
                facts.set(&key, v.clone());
                caps.push(Capability::new(&format!("host.{key}"), CapState::Observable, v, "platform probe"));
            }
            None => caps.push(Capability::new(&format!("host.{key}"), CapState::Unknown, Value::Null, "not probed or probe failed")),
        }
    }
    let code = code_digest.map(str::to_string).or_else(self_digest);
    facts.set("coordinator_digest", code.clone());
    // What this milestone may change: its own workers and explicitly selected applications only.
    caps.push(Capability::new("process.own_worker_threads", CapState::Controllable, Value::Null,
                              "thread count of coordinator-owned workers"));
    for name in ["os.power_profile", "os.cpu_governor", "os.scheduler_priority_global", "os.kernel_parameters"] {
        caps.push(Capability::new(name, CapState::Unsupported, Value::Null,
                                  "privileged system change: needs a separately authorised adapter"));
    }
    caps.push(Capability::new("energy.measurement", CapState::Unknown, Value::Null, "no energy probe implemented"));
    caps.push(Capability::new("temperature.measurement", CapState::Unknown, Value::Null, "no thermal probe implemented"));
    let digest = sha256::hex_digest(facts.dump().as_bytes());
    let class = format!("{}-{}-{}-{}", std::env::consts::OS, std::env::consts::ARCH,
                        facts.str("cpu_model").unwrap_or("unknown-cpu").replace(' ', "_"),
                        logical_cpus().map_or("?".into(), |n| n.to_string()));
    (Identity { facts, digest, hardware_class: class }, caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_stable_and_binds_the_code_digest() {
        let (a, caps) = detect(Some("code-a"));
        let (b, _) = detect(Some("code-a"));
        let (c, _) = detect(Some("code-b"));
        assert_eq!(a.digest, b.digest);
        assert_ne!(a.digest, c.digest);
        assert_eq!(a.digest.len(), 64);
        assert_eq!(a.facts.boolean("test_faults_build"), Some(cfg!(feature = "test-faults")), "a fault build is its own identity");
        let state = |n: &str| caps.iter().find(|c| c.name == n).map(|c| c.state);
        assert_eq!(state("os.power_profile"), Some(CapState::Unsupported));
        assert_eq!(state("energy.measurement"), Some(CapState::Unknown));
        assert_eq!(state("process.own_worker_threads"), Some(CapState::Controllable));
        for c in &caps {
            assert_eq!(CapState::parse(c.state.name()), Some(c.state));
        }
    }
}
