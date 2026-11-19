"""Identify the running llama-server and its loaded llama.cpp libraries for a stock comparison."""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

# Built from the unmodified pinned source and linked against the official v0.3.16-cu124 wheel.
BUNDLED = {
    'schema': 1,
    'upstream': 'https://github.com/ggml-org/llama.cpp',
    'commit': '4227c9be4268ac844921b90f31595f81236bd317',
    'source_tree_clean': True,
    'patches': [],
    'executable_sha256': 'b60f3323c3dd015e1caa881f98295e39e5590791ba8f3acafef8fdfe6f6a56e8',
    'library_sha256': {
        'libggml-base.so': '0d0116b5cd5ee7e5aba15a22254821b2a86c0dcbf7b0d876d65508d8949fe746',
        'libggml-cpu.so': '93f15e7af98cbf7be03f175f9cca50eae6eb1ee0b670968ca386eccce632eb52',
        'libggml-cuda.so': 'c7804984915f63cff92c5398369837fec92bfdaa86a57c70125b99d4530955ab',
        'libggml.so': '3c9078add7a74e8af111db9d18be91e9dbeb6fb6391ee11c8fcdfdbec3fa5f27',
        'libllama.so': '34d04d43475a61e309914b43d78fbadb24caf8d913c2806168fa2a69c6e11c93',
        'libmtmd.so': 'e0374f31a40120b2b334f33ef435fd14d1045a9dbf37fda8c9e2886d87044f70',
    },
}


@functools.lru_cache(maxsize=48)
def _hash(path, inode, size, modified_ns):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path):
    st = path.stat()
    return _hash(path, st.st_ino, st.st_size, st.st_mtime_ns)


def _mapped_images(pid):
    if platform.system() == 'Darwin':
        output = subprocess.check_output(['ps', '-p', str(pid), '-o', 'comm='], text=True, timeout=5)
        exe = Path(output.strip()).resolve(strict=True)
        mapped = subprocess.check_output(['lsof', '-a', '-p', str(pid), '-d', 'txt', '-Fn'], text=True, timeout=8)
        paths = {Path(line[1:]) for line in mapped.splitlines() if line.startswith('n/')}
        if exe not in {p.resolve() for p in paths}:
            raise ValueError('the process image could not be established')
    else:
        proc = Path(f'/proc/{pid}')
        exe = (proc / 'exe').resolve(strict=True)
        paths = {Path(parts[5]) for line in (proc / 'maps').read_text().splitlines()
                 if len(parts := line.split(None, 5)) == 6 and parts[5].startswith('/')}
    libs = [p for p in paths if p.name.startswith(('libggml', 'libllama', 'libmtmd'))]
    if len({p.name for p in libs}) != len(libs):
        raise ValueError('multiple loaded native libraries have the same file name')
    return _fingerprint(exe), {p.name: _fingerprint(p) for p in libs}


def stock_identity(pid, env, version, manifest=None):
    """A caller manifest attests to the source, while hashes and revision are independently checked at runtime."""
    info = {'baseline_stock': False, 'provenance': 'unverified', 'verified_upstream_commit': None}
    try:
        executable, libraries = _mapped_images(pid)
        info.update(executable_sha256=executable, library_sha256=libraries)
        expected = BUNDLED if manifest is None else json.loads(Path(manifest).read_text())
        if manifest is not None:
            info['manifest_sha256'] = _fingerprint(Path(manifest))
        if not isinstance(expected, dict) or expected.get('schema') != 1 or expected.get('upstream') != BUNDLED['upstream']:
            raise ValueError('expected a version 1 upstream llama.cpp manifest')
        if expected.get('patches') != [] or expected.get('source_tree_clean') is not True:
            raise ValueError('manifest declares a modified or unspecified source tree')
        if env.get('LD_PRELOAD') or env.get('DYLD_INSERT_LIBRARIES'):
            raise ValueError('native library preloading is enabled')
        commit = expected.get('commit')
        built = re.search(r'\(([0-9a-f]{7,40})\)', version or '')
        if (not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit)
                or not built or not commit.startswith(built.group(1)) or re.search(r'dirty|patched', version, re.IGNORECASE)):
            raise ValueError('reported server revision does not identify the pinned upstream commit')
        if executable != expected.get('executable_sha256') or libraries != expected.get('library_sha256'):
            raise ValueError('executable or loaded library set differs from the reference manifest')
        info.update(baseline_stock=True, provenance='built-in' if manifest is None else 'caller-pinned',
                    verified_upstream_commit=commit)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        info['reason'] = str(exc)
    return info


def write_manifest(source, executable, libraries, output):
    """Record independently built artifacts from a clean upstream checkout."""
    def git(*args):
        return subprocess.check_output(['git', '-C', source, *args], text=True).strip()

    upstream = BUNDLED['upstream']
    urls = {upstream, 'git@github.com:ggml-org/llama.cpp', 'https://github.com/ggerganov/llama.cpp'}
    remotes = [remote for remote in git('remote').splitlines()
               if git('remote', 'get-url', remote).removesuffix('.git') in urls]
    if not any(git('for-each-ref', '--contains=HEAD', '--format=%(refname)', f'refs/remotes/{remote}/')
               for remote in remotes):
        raise ValueError('expected a commit reachable from an upstream llama.cpp remote-tracking branch')
    if git('status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none'):
        raise ValueError('generate the manifest from a clean source checkout with an external build directory')
    hashes = {}
    for library in libraries:
        path = Path(library).resolve(strict=True)
        if path.name in hashes:
            raise ValueError(f'duplicate library name: {path.name}')
        hashes[path.name] = _fingerprint(path)
    manifest = {'schema': 1, 'upstream': upstream, 'commit': git('rev-parse', 'HEAD'), 'source_tree_clean': True,
                'patches': [], 'executable_sha256': _fingerprint(Path(executable).resolve(strict=True)),
                'library_sha256': hashes}
    # Create the reference once. An existing independent baseline should survive accidental reruns.
    with Path(output).open('x') as dest:
        json.dump(manifest, dest, indent=2, sort_keys=True)
        dest.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fingerprint a separately built clean upstream llama-server.')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--library', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_manifest(args.source, args.executable, args.library, args.output)
