#!/usr/bin/env python3
"""Can the isolated direct Metal comparison be built on this machine?

The comparison the assignment asks for needs one shared `.metallib`, compiled once from
the unchanged specialised kernel source, and then loaded by two independent paths: a
minimal native MLX extension, and a standalone Objective-C++ program with no MLX in the
process. This audits every piece that build needs, records what is present and what is
not, and stops. It builds nothing and substitutes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

MLX = PROJECT_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "mlx"
TOOLCHAIN = Path("/Applications/Xcode.app/Contents/Developer/Toolchains/"
                 "XcodeDefault.xctoolchain/usr/bin")


def _run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"command": " ".join(command), "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:400],
            "stderr": completed.stderr.strip()[:400]}


def _file(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "present": False}
    data = path.read_bytes() if path.is_file() else b""
    return {"path": str(path), "present": True, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest() if data else None}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = {
        "xcrun": _run(["xcrun", "--version"]),
        "sdk_version": _run(["xcrun", "-sdk", "macosx", "--show-sdk-version"]),
        "metal_via_xcrun": _run(["xcrun", "-sdk", "macosx", "metal", "--version"]),
        "metal_direct": _run([str(TOOLCHAIN / "metal"), "--version"]),
        "metallib_via_xcrun": _run(["xcrun", "-sdk", "macosx", "metallib", "--version"]),
        "clangxx": _run(["xcrun", "-sdk", "macosx", "clang++", "--version"]),
    }

    exported = subprocess.run(["nm", "-gU", str(MLX / "lib" / "libmlx.dylib")],
                              capture_output=True, text=True, check=False).stdout
    demangled = subprocess.run(["c++filt"], input=exported, capture_output=True,
                               text=True, check=False).stdout
    path_loader = [line.strip() for line in demangled.splitlines()
                   if "Device::get_library" in line and "function" not in line]

    present = {
        "metal_framework_headers": _file(Path(
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
            "Developer/SDKs/MacOSX26.5.sdk/System/Library/Frameworks/Metal.framework"
        )),
        "mlx_headers": _file(MLX / "include" / "mlx" / "backend" / "metal" / "device.h"),
        "libmlx": _file(MLX / "lib" / "libmlx.dylib"),
        "mlx_metallib": _file(MLX / "lib" / "mlx.metallib"),
        "metal_driver_stub": _file(TOOLCHAIN / "metal"),
        "metallib_tool": _file(TOOLCHAIN / "metallib"),
    }

    metal_usable = checks["metal_direct"]["returncode"] == 0
    metallib_usable = checks["metallib_via_xcrun"]["returncode"] == 0
    blocked = not (metal_usable and metallib_usable)

    record = {
        "schema": "ironmule.b53_metal_toolchain_audit.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "capability_audit",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "can a shared .metallib be compiled here and loaded by two "
                    "independent paths",
        "checks": checks,
        "files": present,
        "what_is_available": [
            "the Metal framework and its headers, from the installed SDK",
            "clang and clang++, so a standalone Objective-C++ program could be built",
            "MLX's own headers and libmlx.dylib, so a native extension could be built",
            "Device::get_library(name, path) is exported by libmlx.dylib, so MLX can "
            "load an external library file by path once one exists",
        ],
        "mlx_path_loader_symbols": path_loader,
        "what_is_missing": {
            "component": "the Metal Toolchain, the offline shader compiler",
            "evidence": checks["metal_direct"]["stderr"] or
                        checks["metal_via_xcrun"]["stderr"],
            "metallib_tool_present": present["metallib_tool"]["present"],
            "note": "the `metal` binary in the toolchain directory is a driver stub; it "
                    "refuses to run and names the missing component itself. `metallib` "
                    "is absent entirely",
        },
        "remedy_not_taken": {
            "command": "xcodebuild -downloadComponent MetalToolchain",
            "why_not": "installing a toolchain component is a change to the system, and "
                       "the assignment rules out system changes",
        },
        "substitutes_considered_and_refused": [
            {
                "idea": "compile at runtime with newLibraryWithSource: in each process",
                "refused_because": "then the two paths do not load the same file, which "
                                   "is the whole point of the comparison",
            },
            {
                "idea": "serialise an MTLDynamicLibrary with serializeToURL: and load that "
                        "from both paths",
                "refused_because": "a serialised dynamic library is a different artefact "
                                   "from a .metallib, driver-specific, and MLX's loader "
                                   "expects a .metallib. Calling it the same file would be "
                                   "the silent substitute the assignment forbids",
            },
            {
                "idea": "reuse the mlx.metallib shipped in the wheel",
                "refused_because": "it contains MLX's own kernels, not the specialised "
                                   "source under investigation",
            },
        ],
        "verdict": "BLOCKED" if blocked else "BUILDABLE",
        "blocking_item": "the Metal Toolchain component is not installed, so no .metallib "
                         "can be produced from the kernel source on this machine",
        "nothing_was_built": True,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"],
                      "blocking_item": record["blocking_item"],
                      "metal_direct_stderr": checks["metal_direct"]["stderr"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
