//! Ownership of measurement processes (backlog OPT3-D (c)).
//!
//! Every adapter call runs under a supervisor (`imopt supervise`, a fresh process per call) that:
//!
//! * holds a shared `flock` on the state's lease file for as long as it lives. The kernel drops the
//!   lock when the supervisor ends, however it ends, so no ownership record can go stale and no pid is
//!   ever stored. A coordinator that (re)starts takes the lease exclusively once before its first
//!   adapter call: while an earlier supervisor still lives (its adapter still running or not yet proven
//!   gone), it waits, and refuses after a bounded wait, so no new measurement starts beside an old one;
//! * starts the adapter as the leader of a new process group. Its descendants stay in that group unless
//!   they leave it on purpose (`setsid`/`setpgid`); the adapter contract forbids that, and IronMule's
//!   own child guard refuses both calls;
//! * watches a lifeline: its stdin is a pipe whose only write end the coordinator holds. The kernel
//!   closes it when the coordinator dies, SIGKILL included, and the supervisor then ends the group:
//!   SIGTERM, up to `GRACE` for the adapter to exit, SIGKILL to the group twice. A normal adapter exit
//!   also ends the group (leftover descendants are leaks, and one holding the adapter's stdout would
//!   otherwise block the call);
//! * signals the group only while its leader is unreaped (`waitid` with `WNOWAIT` sees the exit without
//!   reaping), so the group id cannot belong to anything else; then reaps, then confirms that no member
//!   is left (on Linux it is the subreaper of its descendants and reaps their zombies). Only then does it
//!   exit and release the lease; an unconfirmed group keeps the lease held.
//!
//! Deadline (fixed in OPT3-D (c)): after the coordinator dies, every process of its adapter groups that
//! SIGKILL can end is gone within `DEADLINE` (2 s grace, then SIGKILL); a member that does not die (stuck
//! in the kernel) keeps the supervisor confirming and the lease held, so nothing new starts (fail-closed),
//! with a warning after `CONFIRM`. Supported: macOS and Linux (`cfg(unix)`; tested on macOS arm64 and
//! Linux x86_64). Elsewhere adapters run unsupervised and the guarantee does not hold. Not covered: a
//! supervisor killed with SIGKILL itself, and descendants that leave the adapter's process group.

use std::path::Path;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

pub const GRACE: Duration = Duration::from_secs(2);
pub const CONFIRM: Duration = Duration::from_secs(3);
pub const DEADLINE: Duration = Duration::from_secs(10);
pub const GATE_WAIT: Duration = Duration::from_secs(15);

#[cfg(unix)]
mod sys {
    extern "C" {
        pub fn kill(pid: i32, sig: i32) -> i32;
        pub fn flock(fd: i32, op: i32) -> i32;
        pub fn waitid(idtype: i32, id: u32, info: *mut u64, options: i32) -> i32;
        pub fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        #[cfg(target_os = "linux")]
        pub fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        pub fn signal(sig: i32, handler: usize) -> usize;
        #[cfg(target_os = "linux")]
        pub fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
        #[cfg(not(target_os = "linux"))]
        pub fn poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> i32;
    }
    #[repr(C)]
    pub struct PollFd {
        pub fd: i32,
        pub events: i16,
        pub revents: i16,
    }
    pub const POLLIN: i16 = 0x1;
    pub const SIGHUP: i32 = 1;
    pub const SIGINT: i32 = 2;
    pub const SIGTERM: i32 = 15;
    pub const SIGKILL: i32 = 9;
    pub const ESRCH: i32 = 3;
    pub const LOCK_SH: i32 = 1;
    pub const LOCK_EX: i32 = 2;
    pub const LOCK_NB: i32 = 4;
    pub const LOCK_UN: i32 = 8;
    pub const P_PID: i32 = 1;
    pub const WNOHANG: i32 = 1;
    pub const WEXITED: i32 = 4;
    #[cfg(target_os = "linux")]
    pub const WNOWAIT: i32 = 0x0100_0000;
    #[cfg(not(target_os = "linux"))]
    pub const WNOWAIT: i32 = 0x20;
    /// Byte offset of `si_pid` in `siginfo_t` (64-bit Linux: three ints and padding; Darwin: three ints).
    #[cfg(target_os = "linux")]
    pub const SI_PID: usize = 16;
    #[cfg(not(target_os = "linux"))]
    pub const SI_PID: usize = 12;
}

/// True if a process group with this id still has a member we can see (on macOS zombies may not count;
/// it is only asked after the leader is reaped).
#[cfg(unix)]
pub fn group_exists(pgid: u32) -> bool {
    let r = unsafe { sys::kill(-(pgid as i32), 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() != Some(sys::ESRCH)
}

#[cfg(unix)]
fn signal_group(pgid: u32, sig: i32) {
    unsafe { sys::kill(-(pgid as i32), sig) };
}

/// The child `pid` has exited and is not reaped yet (it stays a zombie, so its pid and group id stay ours).
#[cfg(unix)]
pub fn exited_unreaped(pid: u32) -> bool {
    let mut info = [0u64; 32];
    let r = unsafe { sys::waitid(sys::P_PID, pid, info.as_mut_ptr(), sys::WEXITED | sys::WNOHANG | sys::WNOWAIT) };
    if r != 0 {
        return false;
    }
    let bytes: &[u8; 256] = unsafe { &*(info.as_ptr() as *const [u8; 256]) };
    i32::from_ne_bytes(bytes[sys::SI_PID..sys::SI_PID + 4].try_into().unwrap()) != 0
}

/// A held `flock` on the lease file; released by the kernel when the holder ends.
pub struct Lease {
    _file: std::fs::File,
}

#[cfg(unix)]
fn open_lease(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new().create(true).read(true).write(true).open(path)
}

/// Hold the lease shared (blocking): one per supervisor.
#[cfg(unix)]
pub fn lease_shared(path: &Path) -> std::io::Result<Lease> {
    use std::os::unix::io::AsRawFd;
    let file = open_lease(path)?;
    if unsafe { sys::flock(file.as_raw_fd(), sys::LOCK_SH) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(Lease { _file: file })
}

/// Before a coordinator's first adapter call: no supervisor of an earlier run may still hold the lease.
/// Waits up to `wait`, then refuses (its processes are still running or not proven gone).
#[cfg(unix)]
pub fn await_quiescence(path: &Path, wait: Duration) -> Result<(), String> {
    use std::os::unix::io::AsRawFd;
    let file = open_lease(path).map_err(|e| format!("lease {}: {e}", path.display()))?;
    let t0 = Instant::now();
    let mut told = false;
    loop {
        if unsafe { sys::flock(file.as_raw_fd(), sys::LOCK_EX | sys::LOCK_NB) } == 0 {
            unsafe { sys::flock(file.as_raw_fd(), sys::LOCK_UN) };
            return Ok(());
        }
        if t0.elapsed() >= wait {
            return Err(format!("measurement processes of an earlier run of this state are still running or not proven \
                                gone after {:.0} s (lease {} held); no new measurement starts", wait.as_secs_f64(), path.display()));
        }
        if !told {
            eprintln!("imopt: waiting for the measurement processes of an earlier run to end");
            told = true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(not(unix))]
pub fn await_quiescence(_path: &Path, _wait: Duration) -> Result<(), String> {
    Ok(())
}

/// The supervisor: run `argv` as a new process group under the lease, forward its stdout, end the group
/// when it exits or when the lifeline (stdin) closes. Returns the exit code to use.
#[cfg(unix)]
static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// SIGTERM, SIGINT or SIGHUP to the supervisor end its adapter group first, then the supervisor
/// (a handler, not SIG_IGN: the adapter starts with default dispositions). SIGKILL cannot be caught: a
/// killed supervisor leaves its group to run and releases the lease, which this design does not cover.
#[cfg(unix)]
extern "C" fn on_signal(_: i32) {
    GONE.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The lifeline has already closed (the coordinator died before we could watch it).
#[cfg(unix)]
fn lifeline_closed() -> bool {
    let mut fd = sys::PollFd { fd: 0, events: sys::POLLIN, revents: 0 };
    let r = unsafe { sys::poll(&mut fd, 1, 0) };
    // The coordinator never writes: readable, hung up, an error or an invalid fd all mean it is not there.
    r != 0 || fd.revents != 0
}

#[cfg(unix)]
pub fn supervise(lease: &Path, argv: &[String]) -> i32 {
    use std::io::{Read, Write};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;

    for sig in [sys::SIGTERM, sys::SIGINT, sys::SIGHUP] {
        unsafe { sys::signal(sig, on_signal as extern "C" fn(i32) as usize) };
    }
    let _lease = match lease_shared(lease) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("imopt supervise: lease {}: {e}", lease.display());
            return 125;
        }
    };
    // Checked only now that we hold the lease: a coordinator that died before it, and a resume that
    // passed the gate in that moment, must never meet an adapter started afterwards.
    if lifeline_closed() || GONE.load(Ordering::SeqCst) {
        return 125;
    }
    #[cfg(target_os = "linux")]
    unsafe {
        sys::prctl(36, 1, 0, 0, 0); // PR_SET_CHILD_SUBREAPER: orphaned descendants' zombies come to us
    }
    // The lifeline: EOF (or an error) on stdin means the coordinator is gone.
    std::thread::spawn(move || {
        let mut sink = [0u8; 64];
        let mut stdin = std::io::stdin();
        while matches!(stdin.read(&mut sink), Ok(n) if n > 0) {}
        GONE.store(true, Ordering::SeqCst);
    });
    let Some((program, rest)) = argv.split_first() else {
        eprintln!("imopt supervise: no adapter command");
        return 125;
    };
    let mut child = match Command::new(program).args(rest).process_group(0).stdin(Stdio::null()).stdout(Stdio::piped())
        .stderr(Stdio::inherit()).spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("imopt supervise: cannot start {program}: {e}");
            return 126;
        }
    };
    let pgid = child.id();
    let mut out = child.stdout.take().unwrap();
    let copier = std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        let mut stdout = std::io::stdout();
        while let Ok(n) = out.read(&mut buf) {
            if n == 0 || stdout.write_all(&buf[..n]).and_then(|_| stdout.flush()).is_err() {
                break;
            }
        }
    });
    // Wait for the adapter to exit (not reaped) or the coordinator to die.
    let abandoned = loop {
        if exited_unreaped(pgid) {
            break false;
        }
        if GONE.load(Ordering::SeqCst) {
            break true;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    if abandoned {
        signal_group(pgid, sys::SIGTERM);
        let t0 = Instant::now();
        while t0.elapsed() < GRACE && !exited_unreaped(pgid) {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    // The leader is unreaped (alive or a zombie) here, so `pgid` is still this group's id.
    signal_group(pgid, sys::SIGKILL);
    std::thread::sleep(Duration::from_millis(20));
    signal_group(pgid, sys::SIGKILL); // a member forked during the first one
    let status = child.wait().ok();
    let t0 = Instant::now();
    let mut warned = false;
    while group_exists(pgid) {
        // Reap zombies that came to us as subreaper (Linux); never signal the group again: it is unpinned now.
        let mut st = 0;
        while unsafe { sys::waitpid(-1, &mut st, sys::WNOHANG) } > 0 {}
        if t0.elapsed() > CONFIRM && !warned {
            eprintln!("imopt supervise: process group {pgid} not proven gone; the lease stays held");
            warned = true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = copier.join();
    match status {
        Some(s) => s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0)),
        None => 125,
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn a_child_exit_is_seen_without_reaping_it() {
        let mut c = std::process::Command::new("sh").args(["-c", "exit 7"]).spawn().unwrap();
        let t0 = Instant::now();
        while !exited_unreaped(c.id()) {
            assert!(t0.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(exited_unreaped(c.id()), "still unreaped");
        assert_eq!(c.wait().unwrap().code(), Some(7), "the status is still there to reap");
    }

    #[test]
    fn the_lease_gate_waits_for_a_holder_and_opens_when_it_ends() {
        let dir = std::env::temp_dir().join(format!("imopt-lease-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("measure.lease");
        let held = lease_shared(&path).unwrap();
        assert!(await_quiescence(&path, Duration::from_millis(200)).unwrap_err().contains("still running"));
        drop(held);
        assert!(await_quiescence(&path, Duration::from_millis(200)).is_ok());
    }
}
