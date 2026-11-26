"""Run an untrusted native trial as a disposable unprivileged Linux worker on an owned GPU pod."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes as C
import errno
import json
import os
from pathlib import Path
import platform
import resource
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading

ALLOWED_ENV = {"PATH", "LD_LIBRARY_PATH", "LLAMA_CPP_LIB_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
               "CUDA_HOME", "CUDA_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LANG", "LC_ALL"}


def immutable(path, *, tree=False):
    """Check the named path, ancestors and every resolved symlink target, before dropping uid."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("immutable inputs must be absolute")
    visited = set()

    def check(p, descend):
        real = p.resolve(strict=True)
        for item in {real, p, *real.parents, *p.parents}:
            st = item.lstat()
            if st.st_uid != 0 or (not stat.S_ISLNK(st.st_mode) and st.st_mode & 0o022):
                raise ValueError(f"worker input or ancestor is mutable: {item}")
            if item.is_symlink() and item != p:
                check(item.resolve(strict=True), False)
        if descend and real.is_dir() and real not in visited:
            visited.add(real)
            for part in real.iterdir():
                check(part, True)
    check(path, tree)
    return path.resolve(strict=True)


def filter_syscalls():
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("unverified seccomp architecture")
    lib = C.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [C.c_uint32]
    lib.seccomp_init.restype = C.c_void_p
    lib.seccomp_release.argtypes = [C.c_void_p]
    lib.seccomp_load.argtypes = [C.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint, C.c_void_p]

    class Comparison(C.Structure):
        _fields_ = [("arg", C.c_uint), ("op", C.c_int), ("a", C.c_uint64), ("b", C.c_uint64)]

    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("cannot initialize seccomp")
    def deny(name, error=errno.EPERM, comparison=None):
        nr = lib.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:
            return  # no such syscall on the running architecture/libseccomp
        args = C.byref(comparison) if comparison is not None else None
        if lib.seccomp_rule_add_array(ctx, 0x00050000 | error, nr, int(args is not None), args) != 0:
            raise RuntimeError(f"cannot restrict syscall {name}")

    try:
        for name in ("socket", "socketpair", "connect", "accept", "accept4", "bind", "listen", "sendto",
                     "sendmsg", "sendmmsg", "recvmsg", "recvmmsg", "io_uring_setup", "io_uring_register",
                     "io_uring_enter", "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd",
                     "keyctl", "add_key", "request_key", "bpf", "perf_event_open", "mount", "umount2",
                     "setns", "unshare", "chroot", "pivot_root", "setsid", "setpgid", "open_by_handle_at"):
            deny(name)
        # POSIX compiler children and CUDA threads may run in this process group, with inherited restrictions.
        deny("clone3", errno.ENOSYS)
        for flag in (0x80, 0x20000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000):
            deny("clone", comparison=Comparison(0, 7, flag, flag))  # SCMP_CMP_MASKED_EQ
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError("cannot install worker seccomp filter")
    finally:
        lib.seccomp_release(ctx)


def restrict_files(spec):
    """Landlock ABI 3+ with truncate mediation. Paths outside the allowlist receive no content access."""
    libc = C.CDLL(None, use_errno=True)
    syscall = libc.syscall
    if syscall(444, 0, 0, 1) < 3:  # landlock_create_ruleset, VERSION
        raise RuntimeError("worker requires Linux Landlock ABI 3 or newer")
    execute, write, read_file, read_dir = 1, 2, 4, 8
    handled = (1 << 15) - 1
    rules = C.c_uint64(handled)
    fd = syscall(444, C.byref(rules), C.sizeof(rules), 0)
    if fd < 0:
        raise OSError(C.get_errno(), "cannot create Landlock ruleset")

    class PathRule(C.Structure):
        _pack_ = 1
        _fields_ = [("access", C.c_uint64), ("parent_fd", C.c_int32)]

    def allow(path, rights):
        path = Path(path)
        if not path.exists():
            return
        if not path.is_dir():
            rights &= execute | write | read_file | (1 << 14)
        leaf = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = PathRule(rights, leaf)
            if syscall(445, fd, 1, C.byref(rule), 0) < 0:
                raise OSError(C.get_errno(), f"cannot bind Landlock path {path}")
        finally:
            os.close(leaf)

    try:
        ro = execute | read_file | read_dir
        for name in ["/bin", "/lib", "/lib64", "/usr", "/sys", "/etc/ld.so.cache", "/etc/ld.so.conf",
                     "/etc/ld.so.conf.d", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/localtime",
                     "/etc/os-release", "/etc/alternatives", "/proc/self", "/proc/thread-self", "/proc/cpuinfo",
                     "/proc/meminfo", "/proc/stat", "/proc/version", "/proc/uptime", "/proc/mounts",
                     "/proc/sys", "/proc/driver/nvidia", "/proc/devices", spec["fixture"], *spec["readonly"]]:
            allow(name, ro)
        for dev in [Path("/dev/null"), Path("/dev/zero"), Path("/dev/urandom"), Path("/dev/random"),
                    *Path("/dev").glob("nvidia*")]:
            allow(dev, read_file | write | (read_dir if dev.is_dir() else 0))
        # Within the worker-owned private directory allow ordinary files and compiler outputs.
        allow(spec["output"], handled & ~((1 << 6) | (1 << 9) | (1 << 11)))
        if syscall(446, fd, 0) < 0:  # landlock_restrict_self
            raise OSError(C.get_errno(), "cannot activate Landlock restrictions")
    finally:
        os.close(fd)


def restrict(uid, gid, timeout):
    libc = C.CDLL(None, use_errno=True)
    def prctl(option, value):
        if libc.prctl(option, C.c_ulong(value), 0, 0, 0) != 0:
            raise OSError(C.get_errno(), "prctl")
    prctl(38, 1)  # PR_SET_NO_NEW_PRIVS
    for cap in range(int(Path("/proc/sys/kernel/cap_last_cap").read_text()) + 1):
        prctl(24, cap)  # PR_CAPBSET_DROP
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    for kind, bound in [(resource.RLIMIT_CORE, 0), (resource.RLIMIT_CPU, 4 * timeout + 2),
                        (resource.RLIMIT_NOFILE, 512), (resource.RLIMIT_NPROC, 256),
                        (resource.RLIMIT_FSIZE, 512 << 20)]:
        hard = resource.getrlimit(kind)[1]
        n = min(hard, bound) if hard != resource.RLIM_INFINITY else bound
        resource.setrlimit(kind, (n, n))
    filter_syscalls()


def denied(fn):
    try:
        result = fn()
    except PermissionError:
        return
    if hasattr(result, "close"):
        result.close()
    raise RuntimeError("isolation preflight unexpectedly permitted a forbidden operation")


def child(fd, spec_fd):
    # The preflight protocol has its own one-shot descriptor. Stdin belongs exclusively
    # to the server's data protocol, even if a request arrives before worker startup.
    with os.fdopen(spec_fd) as stream:
        spec = json.load(stream)
    output = Path(spec["output"])
    os.umask(0o077)
    restrict(spec["uid"], spec["gid"], spec["timeout"])
    restrict_files(spec)
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    caps = {k: int(status[k], 16) for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")}
    if (any(caps.values()) or os.getresuid() != (spec["uid"],) * 3 or os.getresgid() != (spec["gid"],) * 3
            or os.getgroups() or int(status["NoNewPrivs"]) != 1 or int(status["Seccomp"]) != 2):
        raise RuntimeError("incorrect worker privilege state")
    for fam in (socket.AF_UNIX, socket.AF_INET, socket.AF_INET6):
        denied(lambda: socket.socket(fam, socket.SOCK_STREAM))
    denied(socket.socketpair)
    denied(lambda: open(spec["sentinel"], "rb"))
    denied(lambda: open(spec["unlisted"], "rb"))
    denied(lambda: open("/proc/1/environ", "rb"))
    denied(lambda: os.listdir("/root"))
    for name in spec["secret_paths"]:
        p = Path(name)
        denied(lambda: os.listdir(p) if p.is_dir() else p.open("rb"))
    for name in spec["readonly"]:
        p = Path(name)
        if not os.access(p, os.R_OK) or os.access(p, os.W_OK):
            raise RuntimeError(f"incorrect access to trial input {p}")
        if p.is_file():
            denied(lambda: p.open("r+b"))
    # The root-owned fixture is world-readable; the process attempts to open it for writing.
    with open(spec["fixture"], "rb") as fixture:
        if fixture.read() != b"reference":
            raise RuntimeError("unreadable reference fixture")
    denied(lambda: open(spec["fixture"], "r+b"))
    with tempfile.TemporaryFile(dir=output) as writable:
        writable.write(b"ok")
    os.chdir(spec["cwd"])
    proof = {"uid": os.getuid(), "gid": os.getgid(), "capabilities": caps, "no_new_privs": True,
             "seccomp": 2, "unlisted_file_denied": True, "network_denied": True, "root_secrets_denied": True, "fixture_write_denied": True,
             "output_writable": True, "immutable_inputs": spec["readonly"]}
    os.write(fd, json.dumps(proof).encode() + b"\n")
    os.close(fd)
    if spec["argv"]:
        os.execvpe(spec["argv"][0], spec["argv"], spec["env"])


@contextmanager
def _worker(argv, *, env, cwd, output, readonly, timeout, uid, gid, secret_paths, stdin, stdout, stderr, text):
    if (os.geteuid() != 0 or not sys.platform.startswith("linux") or not 1000 <= uid < 65534
            or not 1000 <= gid < 65534 or not 1 <= timeout <= 3600):
        raise RuntimeError("worker needs an owned Linux root broker and a dedicated unprivileged uid/gid")
    if not readonly:
        raise ValueError("name the immutable source, model and evaluator")
    output = Path(output).resolve(strict=True)
    parent = immutable(output.parent)
    st = output.stat()
    if not output.is_dir() or st.st_uid != uid or st.st_gid != gid or stat.S_IMODE(st.st_mode) != 0o700:
        raise ValueError("output needs worker ownership and mode 0700")
    cwd = Path(cwd).resolve(strict=True)
    if cwd != output:
        immutable(cwd)
    inputs = [str(immutable(p, tree=True)) for p in readonly]
    protected = [str(Path(p).resolve(strict=True)) for p in secret_paths]
    for key in env:
        if key not in ALLOWED_ENV and not key.startswith(("IRONMULE_", "GGML_", "GGML_CUDA_")):
            raise ValueError(f"unexpected environment key {key}")
        if any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")):
            raise ValueError(f"credential-like environment key {key}")
    clean = {"PATH": "/usr/bin:/bin", "LANG": "C", **env, "HOME": str(output), "TMPDIR": str(output),
             "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_CACHE_PATH": str(output / ".cuda")}
    launcher = immutable(Path(__file__).resolve())
    python = immutable(Path(sys.executable).resolve())
    with tempfile.TemporaryDirectory(dir=parent, prefix=".boundary-") as temp:
        boundary = Path(temp)
        boundary.chmod(0o711)
        sentinel = boundary / "private"
        sentinel.write_bytes(os.urandom(32))
        sentinel.chmod(0o600)
        unlisted = boundary / "unlisted"
        unlisted.write_bytes(b"outside")
        unlisted.chmod(0o444)
        fixture = boundary / "fixture"
        fixture.write_bytes(b"reference")
        fixture.chmod(0o444)
        spec = {"argv": [os.fspath(a) for a in argv], "cwd": str(cwd), "output": str(output), "uid": uid, "gid": gid,
                "timeout": int(timeout), "readonly": inputs, "secret_paths": protected, "env": clean,
                "sentinel": str(sentinel), "unlisted": str(unlisted), "fixture": str(fixture)}
        with tempfile.TemporaryFile(dir=boundary) as control:
            control.write(json.dumps(spec).encode())
            control.seek(0)
            read_fd, write_fd = os.pipe()
            try:
                proc = subprocess.Popen([str(python), "-I", "-S", str(launcher), "_child",
                                         str(write_fd), str(control.fileno())],
                                        stdin=stdin, stdout=stdout, stderr=stderr, text=text, encoding="utf-8" if text else None,
                                        errors="replace" if text else None, bufsize=1 if text else -1, env={"PATH": "/usr/bin:/bin", "LANG": "C"},
                                        cwd="/", pass_fds=(write_fd, control.fileno()), start_new_session=True)
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            def terminate_group():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            deadline = threading.Timer(timeout, terminate_group)
            deadline.daemon = True
            deadline.start()
            try:
                if not select.select([read_fd], [], [], timeout)[0]:
                    raise TimeoutError("restricted worker preflight timed out")
                # A preflight receipt can exceed one pipe write. The child closes the
                # descriptor before exec, so reading to EOF never waits on the server.
                with os.fdopen(read_fd, "rb", closefd=False) as pipe:
                    raw = pipe.read(65536)
                if not raw:
                    raise RuntimeError("restricted worker preflight failed")
                proof = json.loads(raw)
                if proof["uid"] != uid or proof["gid"] != gid or not proof["network_denied"]:
                    raise RuntimeError("incomplete restricted worker preflight")
                yield proc, proof
            finally:
                os.close(read_fd)
                deadline.cancel()
                deadline.join()
                terminate_group()
                proc.wait()
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()


@contextmanager
def open_worker(argv, *, env, cwd, output, readonly, timeout=300, uid=60000, gid=60000, secret_paths=(), stderr=None, text=True):
    """Yield an isolated pipe-protocol worker after preflight, with a hard group deadline."""
    if not argv:
        raise ValueError("interactive worker requires an executable")
    with _worker(argv, env=env, cwd=cwd, output=output, readonly=readonly, timeout=timeout, uid=uid, gid=gid,
                 secret_paths=secret_paths, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                 stderr=subprocess.DEVNULL if stderr is None else stderr, text=text) as (proc, _):
        yield proc


def run(argv, *, env, cwd, output, readonly, timeout=300, uid=60000, gid=60000, secret_paths=()):
    """Capture bounded combined output after the same preflight as the interactive worker."""
    with tempfile.TemporaryFile() as log:
        try:
            with _worker(argv, env=env, cwd=cwd, output=output, readonly=readonly, timeout=timeout, uid=uid, gid=gid,
                         secret_paths=secret_paths, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, text=False) as (proc, proof):
                proc.wait(timeout=timeout)
                returncode = proc.returncode
        except RuntimeError as exc:
            if str(exc) != "restricted worker preflight failed":
                raise
            log.seek(max(0, log.seek(0, os.SEEK_END) - 4000))
            raise RuntimeError(f"{exc}: {log.read().decode(errors='replace')}") from exc
        log.seek(max(0, log.seek(0, os.SEEK_END) - 65536))
        tail = log.read().decode(errors="replace")
        if not argv:
            tail = json.dumps(proof) + "\n" + tail
        return subprocess.CompletedProcess(list(argv), returncode, tail, "")


if __name__ == "__main__":
    if sys.argv[1:2] == ["_child"]:
        child(int(sys.argv[2]), int(sys.argv[3]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--preflight", action="store_true", required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--readonly", nargs="+", type=Path, required=True)
        parser.add_argument("--secret", nargs="*", type=Path, default=[])
        args = parser.parse_args()
        print(run([], env={}, cwd=args.output, output=args.output, readonly=args.readonly,
                  secret_paths=args.secret, timeout=45).stdout.strip())
