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
from uuid import UUID
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


def _cuda_uuids(env):
    # libcuda takes its visibility and order from the environment at initialization. A new
    # interpreter tests the exact child mask even if the compare process already initialized CUDA.
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); "
            "from ironmule.kernel_gen import _visible_cuda_devices; "
            "print(json.dumps([d['uuid'] for d in _visible_cuda_devices()]))")
    probe = subprocess.run([sys.executable, '-I', '-c', code, str(Path(__file__).resolve().parents[1])],
                           env=env, capture_output=True, text=True, timeout=10, check=True)
    return ['GPU-' + str(UUID(hex=value)) for value in json.loads(probe.stdout)]


def _device(env, engine):
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('', '-1') and shutil.which('nvidia-smi', path=env.get('PATH')):
        probe = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], env=env,
                               capture_output=True, text=True, timeout=8, check=False)
        rows = [line.split(', ', 2) for line in probe.stdout.splitlines() if line.strip()]
        if probe.returncode == 0 and rows and probe.stdout.strip() != 'No devices were found':
            if any(len(row) != 3 or not row[0].isascii() or not row[0].isdecimal() for row in rows):
                raise ValueError('unrecognized NVIDIA device inventory')
            first_selector = (visible or '').split(',')[0].strip()
            if first_selector.startswith('GPU-') and sum(uid.startswith(first_selector) for _, uid, _ in rows) != 1:
                raise ValueError('NVIDIA UUID selector is missing or ambiguous in this container')
            devices = _cuda_uuids(env)
            matching = [row for row in rows if devices and row[1].lower() == devices[0].lower()]
            if len(matching) != 1:
                raise ValueError('first visible CUDA device has no unique full GPU UUID in the NVIDIA inventory')
            ordinal, uid, name = matching[0]
            if engine == 'vllm':
                # v0.11.0 casts this selector to int and passes it directly to NVML. The
                # CUDA driver must independently map that same integer to the requested card.
                candidate = {**env, 'CUDA_VISIBLE_DEVICES': ordinal}
                if sum(index == ordinal for index, _, _ in rows) != 1 or _cuda_uuids(candidate) != [devices[0]]:
                    raise ValueError('vLLM numeric selector identifies different devices in CUDA and NVML')
                env['CUDA_VISIBLE_DEVICES'] = ordinal
            else:
                env['CUDA_VISIBLE_DEVICES'] = uid
            return uid, name
    # Other native backends may be active on ROCm, Metal or accelerator-only systems.
    from ironmule_devices import ggml_devices
    devices = ggml_devices()
    if devices:
        if len(devices) > 1:
            raise ValueError('compare needs one visible accelerator per server')
        row = devices[0]
        if row['provider'] == 'cuda' and engine == 'vllm':
            raise ValueError('vLLM needs a verified CUDA and NVML ordinal for the selected NVIDIA device')
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


def vllm_command(args, port, env, model, *, model_options=()):
    """Launch the image-native vLLM server with an explicitly selected local weight format."""
    env.pop('VLLM_API_KEY', None)
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    return [str(args.vllm), 'serve', str(model), '--served-model-name', 'compare',
            '--tokenizer', str(Path(args.tokenizer).resolve()), *model_options, '--generation-config', 'vllm',
            '--max-model-len', str(args.context), '--max-num-seqs', str(_concurrency(args)), '--tensor-parallel-size', '1',
            '--gpu-memory-utilization', str(getattr(args, 'gpu_memory_utilization', 0.88)),
            *(['--enforce-eager'] if getattr(args, 'vllm_enforce_eager', False) else []),
            '--host', '127.0.0.1', '--port', str(port)]


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
        # External benchmark traffic is reserved for the audit and must never trigger a serving holdout.
        env['IRONMULE_ADMISSION_LEARN'] = '0'
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
        return vllm_command(args, port, env, model, model_options=['--hf-config-path', tokenizer, '--quantization', 'gguf'])
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

    def __init__(self, message, stage, code, *, qualification=None, traceback_summary=None, exit_code=None):
        self.code = code
        self.stage = stage
        self.qualification = qualification or []
        self.traceback_summary = traceback_summary
        self.exit_code = exit_code
        super().__init__(f'{message} (stage={stage} reason={code})')

    def report_fields(self):
        fields = {'startup_stage': self.stage, 'startup_reason': self.code}
        if self.qualification:
            fields['qualification'] = self.qualification
        if self.traceback_summary is not None:
            fields['traceback_summary'] = self.traceback_summary
        if self.exit_code is not None:
            fields['exit_code'] = self.exit_code
        return fields

    @classmethod
    def from_log(cls, engine, logpath, message, exit_code=None):
        try:
            with logpath.open('rb') as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 128 * 1024))
                log = stream.read().decode('utf-8', errors='replace')
        except OSError:
            return cls(message, 'unknown', 'log_unavailable', exit_code=exit_code)
        qualification = _gate_attempts(log) if engine == 'ironmule' else []
        stage, code = _classify_log(log, gate_passed=bool(qualification and qualification[-1]['passes']))
        return cls(message, stage, code, exit_code=exit_code, qualification=qualification,
                   traceback_summary=_vllm_traceback(log) if engine == 'vllm' else None)


class StartupTimeout(StartupError, TimeoutError):
    """Preserve timeout semantics alongside the last known startup stage."""


def _gate_attempts(log):
    from ironmule.ggml_gate_diagnostics import sanitize

    attempts = {}
    for line in log.splitlines():
        line = line.lstrip()
        if len(line) > 48_000 or not line.startswith('{"ggml_gate"'):
            continue
        try:
            gate = json.loads(line).get('ggml_gate')
        except (ValueError, RecursionError):
            continue
        if not isinstance(gate, dict) or type(gate.get('passes')) is not bool:
            continue
        mode = gate.get('kernel_mode')
        if mode not in ('generated', 'stock'):
            continue
        result = {'kernel_mode': mode, 'passes': gate['passes']}
        diagnostic = sanitize(gate.get('diagnostic'))
        if diagnostic:
            result['diagnostic'] = diagnostic
        attempts[mode] = result
    return list(attempts.values())


def _vllm_traceback(log):
    """Only emit fixed package names, exception types and numeric source lines."""
    classes = ('ImportError', 'ModuleNotFoundError', 'OSError', 'FileNotFoundError', 'KeyError', 'TypeError',
               'ValueError', 'RuntimeError', 'ValidationError', 'AssertionError', 'AttributeError', 'MemoryError',
               'NotImplementedError', 'OutOfMemoryError', 'BackendCompilerFailed', 'CalledProcessError',
               'HFValidationError', 'OfflineModeIsEnabled', 'PermissionError', 'TimeoutError', 'IndexError',
               'NotADirectoryError', 'FileExistsError', 'JSONDecodeError', 'UnicodeDecodeError', 'PydanticUserError',
               'NVMLError_LibraryNotFound', 'NVMLError_DriverNotLoaded', 'EngineCoreProcError')
    component_paths = ('vllm/entrypoints/', 'vllm/engine/', 'vllm/config/', 'vllm/platforms/',
                       'vllm/model_executor/model_loader/', 'vllm/model_executor/layers/quantization/',
                       'vllm/model_executor/models/', 'vllm/transformers_utils/', 'vllm/compilation/',
                       'vllm/v1/worker/', 'vllm/v1/engine/', 'gguf/', 'transformers/', 'huggingface_hub/',
                       'torch/', 'triton/', 'flashinfer/', 'xformers/', 'pydantic/', 'uvloop/')
    files = ('vllm/config/model.py', 'vllm/engine/arg_utils.py', 'vllm/entrypoints/cli/main.py',
             'vllm/entrypoints/cli/serve.py', 'vllm/entrypoints/openai/api_server.py',
             'vllm/model_executor/model_loader/gguf_loader.py', 'vllm/model_executor/layers/quantization/gguf.py',
             'vllm/model_executor/model_loader/weight_utils.py', 'vllm/model_executor/models/qwen3.py',
             'vllm/transformers_utils/config.py', 'vllm/platforms/cuda.py', 'vllm/v1/worker/gpu_worker.py',
             'vllm/v1/worker/gpu_model_runner.py', 'gguf/gguf_reader.py', 'transformers/configuration_utils.py')
    frames = []
    for path, number in re.findall(r'File "([^"\n]+)", line ([0-9]{1,6})\b', log):
        component = next((part.rstrip('/') for part in component_paths if '/' + part in path), None)
        if component:
            frame = {'component': component, 'line': int(number)}
            known_file = next((f for f in files if path.endswith('/' + f)), None)
            if known_file:
                frame['source'] = known_file
            if frame not in frames:
                frames.append(frame)
    exceptions = re.findall(r'\b(' + '|'.join(classes) + r')(?=[:\n]|$)', log)
    return {'traceback_seen': 'Traceback (most recent call last)' in log,
            'exceptions': list(dict.fromkeys(exceptions))[-8:], 'frames': frames[-8:]}


def _classify_log(log, *, gate_passed=False):
    stages = (
        ('worker', r'"agent":\s*\{'), ('width_selection', r'"ggml_sequences":'),
        ('quality_gate', r'"ggml_gate":'), ('gguf_identified', r'"gguf_model":'),
        ('model_loading', r'Loading model weights|Starting to load model|loading model tensors'),
        ('vllm_engine', r'Initializing a V\d+ LLM engine|EngineCore'),
    )
    stage = next((key for key, pattern in stages if re.search(pattern, log)), 'startup')
    causes = (
        ('missing_mlx', r"No module named ['\"]mlx(?:\.|['\"])"),
        ('missing_gguf', r"No module named ['\"]gguf(?:\.|['\"])"),
        ('missing_accelerator_dependency', r"No module named ['\"](?:vllm|triton|torch|flashinfer|xformers)(?:\.|['\"])"),
        ('missing_python_dependency', r'ModuleNotFoundError|No module named'),
        ('native_symbol_unavailable', r'undefined symbol:|version [`\']GLIBC(?:XX)?_[^\n]+not found'),
        ('incompatible_python_api', r'cannot import name|unexpected keyword argument|operator [^\n]+ does not exist'),
        ('non_numeric_cuda_selector', r"invalid literal for int\(\) with base 10: ['\"](?:GPU|MIG)-"),
        ('device_detection_failed', r'Failed to infer device type|No CUDA GPUs are available|Found no NVIDIA driver'),
        ('cuda_library_unavailable', r'lib(?:cuda|cublas|cudart|nvrtc)[^\s:]*: cannot open shared object'),
        ('cuda_driver_incompatible', r'driver version is insufficient|driver.*too old|CUDA_ERROR_UNSUPPORTED_PTX_VERSION'),
        ('insufficient_free_gpu_memory', r'less than (?:the )?desired GPU memory utilization'),
        ('gpu_out_of_memory', r'CUDA out of memory|OutOfMemoryError|cudaMalloc failed|out of memory on device'),
        ('incompatible_gpu_architecture', r'no kernel image is available|invalid device function|compute capability.*not supported'),
        ('unsupported_gguf', r'unsupported (?:model architecture|quantization)|model type .*not supported|cannot map tensor'),
        ('invalid_model_config', r'Unrecognized model|Unrecognized configuration|Invalid repository ID|Repo(?:sitory)? id must|validation errors? for ModelConfig'),
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
    reason = next((key for key, pattern in causes if not (gate_passed and key == 'quality_gate_refused')
                   and re.search(pattern, log, re.IGNORECASE)), 'unclassified')
    return stage, reason


def open_server(engine, args, repeat, deadline):
    return open_prepared_server(engine, args, repeat, deadline, prepare=prepare_artifacts, command=_command)


@contextlib.contextmanager
def open_prepared_server(engine, args, repeat, deadline, *, prepare, command):
    """Own the lifetime, device binding and health checks of a server launched from verified local artifacts."""
    artifacts = prepare(args)
    workdir = _paths(args)
    env = os.environ.copy()
    if engine == 'vllm' and env.get('IRONMULE_COMPARE_WRAPPED') == '1':
        # Probe and launch vLLM under the same image-native library and Python environment.
        env['PYTHONPATH'] = env.get('IRONMULE_COMPARE_ORIGINAL_PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = env.get('IRONMULE_COMPARE_ORIGINAL_LD_LIBRARY_PATH', '')
    try:
        device, name = _device(env, engine)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise StartupError('unable to verify the selected device', 'device_selection', 'device_selection_unverified') from exc
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    cmd = command(engine, args, port, env, not device.startswith('cpu:'))
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
                    raise StartupError.from_log(engine, logpath, f'{engine} exited with status {code} during startup', code)
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
                raise StartupTimeout.from_log(engine, logpath, f'{engine} startup expired')
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
            if device.startswith('GPU-'):
                info['cuda_visible_devices'] = env['CUDA_VISIBLE_DEVICES']
            if engine == 'vllm':
                info['model_root'] = next(m.get('root') for m in models if m.get('id') == model_id)
            if engine == 'ironmule':
                qualification = _gate_attempts(log)
                if qualification:
                    info['qualification'] = qualification
            yield info
            prepare(args)
        finally:
            _stop(process)
