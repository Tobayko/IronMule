"""Launch local production serving processes for the HTTP comparison, one at a time."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_artifacts(args):
    """Compute the streaming GGUF hash and file-name-bound tokenizer manifest once per invocation."""
    model = Path(args.gguf).resolve(strict=True)
    tokenizer = Path(args.tokenizer).resolve(strict=True)
    if not model.is_file() or not tokenizer.is_dir() or not (tokenizer / 'tokenizer_config.json').is_file():
        raise ValueError('compare needs a GGUF file and a local Hugging Face tokenizer directory')
    paths = [p for p in sorted(tokenizer.rglob('*')) if p.is_file()
             and p.suffix in {'.json', '.jinja', '.model', '.txt', '.tiktoken'}
             and not any(part.startswith('.') for part in p.relative_to(tokenizer).parts)]

    def snapshot():
        stamps = []
        for path in [model, *paths]:
            st = path.stat()
            stamps.append((str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        return stamps

    before = snapshot()
    if getattr(args, '_compare_stamps', None) is None:
        files = {p.relative_to(tokenizer).as_posix(): _sha(p) for p in paths}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        args.gguf_sha256, args.tokenizer_sha256 = _sha(model), digest
        if snapshot() != before:
            raise ValueError('benchmark artifacts changed during hashing')
        args._compare_stamps = before
        args._compare_artifacts = {'gguf_sha256': args.gguf_sha256, 'tokenizer_sha256': digest,
                                   'tokenizer_files': files}
    elif before != args._compare_stamps:
        raise ValueError('GGUF or tokenizer changed between serving arms')
    return args._compare_artifacts


def _placement(engine, device, health, log):
    accelerator = not device.startswith('cpu:')
    if engine == 'ironmule':
        actual = health.get('device')
        return actual == ('gpu' if accelerator else 'cpu'), f'worker health.device={actual}'
    if engine == 'llama_cpp':
        hits = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', log)
        if accelerator:
            return bool(hits and int(hits[-1][0]) > 0), f'GGML offloaded layers={hits[-1] if hits else None}'
        return ('CPU' in log and 'model buffer size' in log and not any(int(a) for a, _ in hits)), 'GGML CPU model buffers, -ngl 0'
    if engine == 'vllm':
        backend = re.search(r'device_config=(?:DeviceConfig\(device=)?[\"\']?(cuda|cpu|rocm|xpu|tpu)', log)
        if not backend:
            backend = re.search(r'Automatically detected platform (cuda|cpu|rocm|xpu|tpu)', log)
        platform_name = backend.group(1) if backend else None
        provider = 'cuda' if device.startswith('GPU-') else device.split(':', 1)[0]
        actual = 'rocm' if platform_name == 'cuda' and re.search(r'platform (?:ROCm|rocm)', log) else platform_name
        return actual == provider, f'vLLM engine backend={actual}'
    return False, 'unidentified runtime'


def _device(env):
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('', '-1') and shutil.which('nvidia-smi'):
        probe = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=8, check=False)
        rows = [line.split(', ', 2) for line in probe.stdout.splitlines()]
        if probe.returncode == 0 and rows and len(rows[0]) == 3:
            target = visible.split(',')[0] if visible else rows[0][1]
            for idx, uuid, name in rows:
                if target in (idx, uuid) or (target.startswith('GPU-') and uuid.startswith(target)):
                    env['CUDA_VISIBLE_DEVICES'] = uuid
                    return uuid, name
            raise ValueError(f'cannot identify visible NVIDIA GPU {target}')
    # Other native backends may be active on ROCm, Metal or accelerator-only systems.
    from ironmule_devices import ggml_devices
    devices = ggml_devices()
    if devices:
        if len(devices) > 1:
            raise ValueError('compare needs one visible accelerator per server')
        row = devices[0]
        return f"{row['provider']}:{platform.node()}:{row['name']}", row['description']
    return f'cpu:{platform.node()}:{platform.machine()}', platform.processor() or platform.machine()


def _json(base, path, timeout, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def _stop(process):
    # Both vLLM and IronMule create children. Terminate the whole owned session even if its leader exited.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def _paths(args):
    if not getattr(args, '_compare_workdir', None):
        logdir = getattr(args, 'log_dir', None)
        if logdir is None:
            args._compare_workdir = Path(tempfile.mkdtemp(prefix='ironmule-compare-'))
        else:
            args._compare_workdir = Path(logdir).resolve()
        args._compare_workdir.mkdir(parents=True, exist_ok=True)
    return args._compare_workdir


def _concurrency(args):
    values = args.concurrency.split(',') if isinstance(args.concurrency, str) else args.concurrency
    return max(int(n) for n in values)


def _command(engine, args, port, env, gpu):
    model, tokenizer = str(Path(args.gguf).resolve()), str(Path(args.tokenizer).resolve())
    address = ['--host', '127.0.0.1', '--port', str(port)]
    context, width = str(args.context), _concurrency(args)
    if engine == 'ironmule':
        state = Path(getattr(args, 'agent_dir', None) or _paths(args) / 'agent').resolve()
        if not (state / 'settings.json').exists():
            from .state import ProductStore
            ProductStore(state).setup('server')
        cmd = [str(args.ironmule)] if getattr(args, 'ironmule', None) else [sys.executable, '-m', 'ironmule_cli']
        env.pop('IRONMULE_COMPARE_LOCAL_KEY', None)
        env['IRONMULE_GGML_SEQ_CTX'] = context
        return [*cmd, 'serve', '--gguf', model, '--state-dir', str(state), '--agent', 'on', '--no-thinking',
                '--api-key-env', 'IRONMULE_COMPARE_LOCAL_KEY', *address]
    if engine == 'llama_cpp':
        from ironmule.pilot import usable_cores
        threads = str(env.get('IRONMULE_GGML_THREADS') or usable_cores())
        return [str(args.llama_server), '-m', model, '--alias', 'compare', '-np', str(width), '-t', threads, '-tb', threads,
                '-c', str(int(context) * width), '-ngl', '999' if gpu else '0',
                '--jinja', '--chat-template-kwargs', '{"enable_thinking":false}', *address]
    if engine == 'vllm':
        # The HF sidecar supplies the architecture config as well as the tokenizer. Weights come only from the GGUF.
        if env.get('IRONMULE_COMPARE_WRAPPED') == '1':
            # In the Runpod bundle vLLM keeps its image's original Torch/CUDA and Python packages.
            env['PYTHONPATH'] = env.get('IRONMULE_COMPARE_ORIGINAL_PYTHONPATH', '')
            env['LD_LIBRARY_PATH'] = env.get('IRONMULE_COMPARE_ORIGINAL_LD_LIBRARY_PATH', '')
        env.pop('VLLM_API_KEY', None)
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        return [str(args.vllm), 'serve', model, '--served-model-name', 'compare', '--tokenizer', tokenizer,
                '--hf-config-path', tokenizer, '--quantization', 'gguf', '--generation-config', 'vllm',
                '--max-model-len', context, '--max-num-seqs', str(width), '--tensor-parallel-size', '1',
                '--gpu-memory-utilization', str(getattr(args, 'gpu_memory_utilization', 0.88)),
                *(['--enforce-eager'] if getattr(args, 'vllm_enforce_eager', False) else []), *address]
    raise ValueError(f'unknown serving engine {engine!r}')


def _health_identity(engine, base, timeout):
    if engine == 'ironmule':
        health = _json(base, '/health', timeout)
        if not health.get('ready') or health.get('ggml', {}).get('gate', {}).get('passes') is not True:
            raise RuntimeError('IronMule has not qualified and loaded the GGUF on this host')
        return health
    if engine == 'llama_cpp' and _json(base, '/health', timeout).get('status') != 'ok':
        raise RuntimeError('llama-server is still loading its model')
    if engine == 'vllm':
        with urlopen(base + '/health', timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError('vLLM is still loading its model')
    return {}


def refresh_identity(engine, server, deadline):
    """Read the agent that actually handled the warmup from worker health after generation."""
    if engine == 'ironmule':
        health = _health_identity(engine, server['base_url'], max(0.05, min(5, deadline - time.monotonic())))
        agent = health.get('ggml', {}).get('agent', {})
        server['agent_status'] = agent
        pilot = health['ggml'].get('pilot') or {}
        server['pilot_status'] = pilot
        server['autonomous'] = bool((agent.get('serving') and agent.get('configs')) or pilot.get('active') is True)
        binding = health['ggml'].get('experiment_binding')
        if binding:
            server['ggml'] = binding
        server['serving_device'] = health.get('device')


class StartupError(RuntimeError):
    """A public failure code and stage from the serving process' bounded, private log."""

    def __init__(self, message, stage, code):
        self.code = code
        self.stage = stage
        super().__init__(f'{message} (stage={stage} reason={code})')


def _startup_detail(logpath):
    """Convert a bounded private log tail to fixed public status codes without returning log contents."""
    try:
        with logpath.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 128 * 1024))
            log = stream.read().decode('utf-8', errors='replace')
    except OSError:
        return 'unknown', 'log_unavailable'
    stages = (
        ('worker', r'"agent":\s*\{'), ('width_selection', r'"ggml_sequences":'),
        ('quality_gate', r'"ggml_gate":'), ('gguf_identified', r'"gguf_model":'),
        ('vllm_engine', r'Initializing a V\d+ LLM engine|EngineCore'),
        ('model_loading', r'Loading model weights|Starting to load model|loading model tensors'),
    )
    stage = next((key for key, pattern in stages if re.search(pattern, log)), 'startup')
    causes = (
        ('missing_mlx', r"No module named ['\"]mlx(?:\.|['\"])"),
        ('missing_gguf', r"No module named ['\"]gguf(?:\.|['\"])"),
        ('missing_accelerator_dependency', r"No module named ['\"](?:vllm|triton|torch|flashinfer|xformers)(?:\.|['\"])"),
        ('missing_python_dependency', r'ModuleNotFoundError|No module named'),
        ('cuda_library_unavailable', r'lib(?:cuda|cublas|cudart|nvrtc)[^\s:]*: cannot open shared object'),
        ('cuda_driver_incompatible', r'driver version is insufficient|driver.*too old|CUDA_ERROR_UNSUPPORTED_PTX_VERSION'),
        ('insufficient_free_gpu_memory', r'less than (?:the )?desired GPU memory utilization'),
        ('gpu_out_of_memory', r'CUDA out of memory|OutOfMemoryError|cudaMalloc failed|out of memory on device'),
        ('incompatible_gpu_architecture', r'no kernel image is available|invalid device function|compute capability.*not supported'),
        ('unsupported_gguf', r'unsupported (?:model architecture|quantization)|model type .*not supported|cannot map tensor'),
        ('invalid_model_config', r'Unrecognized model|Unrecognized configuration|Invalid repository ID|Repository id must'),
        ('missing_weight_files', r'(?:GGUF|weight|model).*(?:file not found|no such file)|FileNotFoundError'),
        ('cuda_graph_or_compiler', r'TritonCompilationError|TorchInductorError|BackendCompilerFailed|\binductor[^\n]*(?:failed|error)|CUDA error[^\n]*captur|CUDAGraph[^\n]*(?:failed|error)'),
        ('collective_initialization', r'NCCL error|NCCL.*failed|ncclSystemError'),
        ('shared_memory_unavailable', r'No space left on device|Bus error|shared memory.*(?:failed|insufficient)'),
        ('cli_argument_rejected', r'unrecognized arguments|No such option|invalid choice'),
        ('permission_denied', r'PermissionError|Permission denied'),
        ('quality_gate_refused', r'GGUF did not pass|"passes":\s*false|no quality-qualified'),
        ('worker_unavailable', r'backend worker|backend(?: is)?[_ ]unavailable|worker failed to start'),
        ('engine_initialization', r'engine core initialization failed|EngineCore failed to start|engine process failed'),
    )
    reason = next((key for key, pattern in causes if re.search(pattern, log, re.IGNORECASE)), 'unclassified')
    return stage, reason


@contextlib.contextmanager
def open_server(engine, args, repeat, deadline):
    artifacts = prepare_artifacts(args)
    workdir = _paths(args)
    env = os.environ.copy()
    device, name = _device(env)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    cmd = _command(engine, args, port, env, not device.startswith('cpu:'))
    logpath = workdir / f'{engine}-r{repeat}-{port}.log'
    started = time.monotonic()
    until = min(deadline, started + args.startup_timeout)
    if started >= until:
        raise TimeoutError(f'budget exhausted before {engine} startup')
    with logpath.open('wb') as logs:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logs, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            while time.monotonic() < until:
                code = process.poll()
                if code is not None:
                    raise StartupError(f'{engine} exited with status {code} during startup', *_startup_detail(logpath))
                try:
                    timeout = max(0.05, min(2, until - time.monotonic()))
                    models = _json(base, '/v1/models', timeout).get('data', [])
                    if models:
                        health = _health_identity(engine, base, timeout)
                        break
                except (URLError, OSError, ValueError, RuntimeError):
                    pass
                time.sleep(min(0.2, max(0, until-time.monotonic())))
            else:
                stage, code = _startup_detail(logpath)
                raise TimeoutError(f'{engine} startup expired (stage={stage} reason={code})')
            model_id = health.get('loaded_model') if engine == 'ironmule' else 'compare'
            if model_id not in [m.get('id') for m in models]:
                raise RuntimeError(f'{engine} did not advertise its requested model')
            if device.startswith('GPU-') and engine == 'ironmule' and health.get('device') != 'gpu':
                raise RuntimeError('IronMule did not load the GGUF on the selected GPU')
            log = logpath.read_text(errors='replace')
            placed, placement_evidence = _placement(engine, device, health, log)
            identity = {}
            version = (models[0].get('meta') or {}).get('version')
            if engine == 'ironmule':
                from ironmule._version import __version__
                version = __version__
            elif engine == 'vllm':
                version = _json(base, '/version', timeout).get('version')
            elif engine == 'llama_cpp':
                # This command only prints build metadata and exits, even while another instance serves.
                output = subprocess.run([cmd[0], '--version'], env=env, capture_output=True, text=True, timeout=5, check=False)
                version = next((s.strip() for s in (output.stdout+'\n'+output.stderr).splitlines()
                                if s.strip().startswith('version:')), None)
                from .compare_stock import stock_identity
                identity = stock_identity(process.pid, env, version, getattr(args, 'stock_manifest', None))
            info = {**artifacts, 'engine': engine, 'base_url': base, 'argv': cmd, 'model_id': model_id,
                    'autonomous': False, 'agent_requested': engine == 'ironmule', 'version': version,
                    'device': device, 'device_name': name, 'serving_device': health.get('device'),
                    'placement_verified': placed, 'placement_evidence': placement_evidence,
                    'provider': 'cuda' if device.startswith('GPU-') else device.split(':',1)[0],
                    'identity': identity, 'baseline_stock': identity.get('baseline_stock'),
                    'cpu_threads': next(iter(re.findall(r'n_threads = (\d+)', log)), None),
                    'prefix_caching': 'engine default',
                    'startup_seconds': time.monotonic() - started, 'log_file': str(logpath),
                    'launcher_executable': shutil.which(cmd[0], path=env.get('PATH')) or cmd[0]}
            yield info
            prepare_artifacts(args)
        finally:
            _stop(process)
