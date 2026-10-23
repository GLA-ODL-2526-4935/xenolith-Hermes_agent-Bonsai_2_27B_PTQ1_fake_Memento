"""Prism process backend for Bonsai; does not use the Gemma C graph or snapshot format."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

MODEL = 'Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf'
MODEL_SHA256 = '1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685'
FIELDS = {'schema_version', 'id', 'backend', 'target_os', 'device', 'context', 'threads',
          'batch', 'ubatch', 'kv_type', 'reasoning_effort', 'max_output_tokens',
          'mtp_tokens', 'mtp_probability', 'batch_invariant', 'min_available_ram_gib',
          'host', 'port', 'status', 'runtime_requirement'}
CPU_OPTIONS = {'threads_batch': None, 'cpu_mask': '', 'cpu_mask_batch': '',
               'poll': 0, 'poll_batch': 0, 'prefix_cache': False,
               'cache_ram_mib': 0, 'adaptive_mtp': False, 'draft_backend_sampling': True,
               'source_policy': False, 'suffix_drafting': False, 'policy_frozen': False,
               'policy_identity': '', 'policy_input': '', 'policy_output': ''}


def settings(c):
    result = {**CPU_OPTIONS, **c}
    if result['threads_batch'] is None:
        result['threads_batch'] = result['threads']
    return result


def load_profile(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate profile field')
            out[key] = value
        return out
    cfg = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs)
    if not FIELDS <= set(cfg) or set(cfg) - FIELDS - CPU_OPTIONS.keys():
        raise ValueError('Missing or unknown profile fields')
    validate(cfg)
    return cfg


def validate(c):
    c = settings(c)
    if c['schema_version'] != 1 or c['backend'] != 'prism-llama.cpp' or c['target_os'] != 'Linux':
        raise ValueError('Only the explicit Linux Prism backend is supported')
    if c['host'] != '127.0.0.1' or c['reasoning_effort'] not in {'off', 'low', 'medium'}:
        raise ValueError('Use loopback and an explicit supported reasoning effort')
    if c['kv_type'] not in {'q4_0', 'q8_0', 'f16'} or type(c['batch_invariant']) is not bool:
        raise ValueError('Invalid cache or batch policy')
    limits = {'context': (512, 262144), 'threads': (1, 64), 'batch': (1, 4096),
              'ubatch': (1, 4096), 'max_output_tokens': (1, 16384),
              'mtp_tokens': (0, 4), 'port': (1024, 65535), 'min_available_ram_gib': (5, 64)}
    for key, (low, high) in limits.items():
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError('Invalid value for ' + key)
    if c['ubatch'] > c['batch'] or c['max_output_tokens'] >= c['context']:
        raise ValueError('Invalid batch or output/context relationship')
    if type(c['threads_batch']) is not int or not 1 <= c['threads_batch'] <= 64:
        raise ValueError('Invalid batch thread count')
    for field in ('prefix_cache', 'adaptive_mtp', 'draft_backend_sampling',
                  'source_policy', 'suffix_drafting', 'policy_frozen'):
        if type(c[field]) is not bool:
            raise ValueError('Expected boolean ' + field)
    for field in ('poll', 'poll_batch'):
        if type(c[field]) is not int or c[field] not in (0, 1):
            raise ValueError('Polling must be 0 or 1')
    for field in ('cpu_mask', 'cpu_mask_batch'):
        mask = c[field]
        if not isinstance(mask, str) or (mask and (len(mask) > 256 or
                any(ch not in '0123456789abcdefABCDEF' for ch in mask) or int(mask, 16) == 0)):
            raise ValueError('Affinity requires a nonzero plain hexadecimal mask')
    if type(c['cache_ram_mib']) is not int or not 0 <= c['cache_ram_mib'] <= 256:
        raise ValueError('Cache budget must be 0..256 MiB')
    if c['cache_ram_mib'] and not c['prefix_cache']:
        raise ValueError('Cache budget requires explicit prefix reuse')
    if c['mtp_tokens'] and c['ubatch'] < c['mtp_tokens'] + 1:
        raise ValueError('Keep one complete verification cycle in a microbatch')
    if c['adaptive_mtp'] and (not c['mtp_tokens'] or c['mtp_probability'] != 0 or c['device'] != 'none'):
        raise ValueError('Adaptive CPU MTP requires CPU, MTP enabled and zero threshold')
    if c['adaptive_mtp'] and 'cpu-mtp-v1' not in c['runtime_requirement']:
        raise ValueError('Adaptive mode requires an explicitly patched cpu-mtp-v1 runtime')
    for field in ('policy_identity', 'policy_input', 'policy_output'):
        if not isinstance(c[field], str) or '\x00' in c[field]:
            raise ValueError('Invalid selector path')
    if c['source_policy']:
        if not c['mtp_tokens'] or c['adaptive_mtp'] or 'intel-source-profile-v2' not in c['runtime_requirement']:
            raise ValueError('Source policy requires MTP and the portable patched runtime; disable old adaptive MTP')
        if any(c[f] for f in ('policy_identity', 'policy_input', 'policy_output')):
            if not c['policy_identity'] or not (c['policy_input'] or c['policy_output']):
                raise ValueError('Profile IO requires identity and input/output')
        if c['policy_frozen'] and (not c['policy_input'] or c['policy_output']):
            raise ValueError('Frozen policy requires input only')
    elif any(c[f] for f in ('suffix_drafting', 'policy_frozen', 'policy_identity', 'policy_input', 'policy_output')):
        raise ValueError('Selector options require source_policy')
    p = c['mtp_probability']
    if type(p) not in (int, float) or not 0 <= p <= 1:
        raise ValueError('Invalid draft probability')
    device = c['device']
    if not isinstance(device, str) or (device != 'none' and not
            (device.startswith('Vulkan') and device[6:].isdigit())):
        raise ValueError('Select none for CPU or an explicitly enumerated Vulkan device')


def request_defaults(c):
    c = settings(c)
    on = c['reasoning_effort'] != 'off'
    result = {'model': 'bonsai-2-27b', 'temperature': 0, 'top_k': 1, 'top_p': 1,
            'min_p': 0, 'repeat_penalty': 1, 'seed': 3407,
            'samplers': ['top_k', 'temperature'],
            'max_tokens': c['max_output_tokens'], 'reasoning_effort': c['reasoning_effort'],
            'chat_template_kwargs': {'enable_thinking': on, 'reasoning_effort': c['reasoning_effort']}}
    result['cache_prompt'] = c['prefix_cache']
    if c['prefix_cache']:
        result['id_slot'] = 0
    return result


def make_plan(c, engine, model):
    validate(c)
    c = settings(c)
    if Path(model).name != MODEL:
        raise ValueError('This profile requires the pinned Bonsai 27B Lean filename')
    cpu = c['device'] == 'none'
    cmd = [str(engine), '-m', str(model), '--alias', 'bonsai-2-27b', '--host', c['host'],
           '--port', str(c['port']), '--device', c['device'], '-ngl', '0' if cpu else '99',
           '-c', str(c['context']), '-np', '1', '-t', str(c['threads']),
           '-tb', str(c['threads_batch']), '--poll', str(c['poll']),
           '--poll-batch', str(c['poll_batch']),
           '-b', str(c['batch']), '-ub', str(c['ubatch']), '-fa', 'auto',
           '-ctk', c['kv_type'], '-ctv', c['kv_type'], '--fit', 'off', '--jinja',
           '--no-context-shift', '--cache-ram', str(c['cache_ram_mib']),
           '--cache-prompt' if c['prefix_cache'] else '--no-cache-prompt', '--no-webui',
           '--reasoning', 'off' if c['reasoning_effort'] == 'off' else 'on',
           '--reasoning-effort', c['reasoning_effort'], '-n', str(c['max_output_tokens']),
           '--temp', '0', '--top-k', '1', '--top-p', '1', '--min-p', '0',
           '--repeat-penalty', '1', '--seed', '3407', '--samplers', 'top_k;temperature',
           '--spec-type', 'draft-mtp' if c['mtp_tokens'] else 'none']
    if c['mtp_tokens']:
        cmd += ['--spec-draft-n-max', str(c['mtp_tokens']), '--spec-draft-n-min', '0',
                '--spec-draft-p-min', str(c['mtp_probability'])]
        cmd += ['--spec-draft-threads', str(c['threads']),
                '--spec-draft-threads-batch', str(c['threads_batch']),
                '--spec-draft-poll', str(c['poll']), '--spec-draft-poll-batch', str(c['poll_batch']),
                '--spec-draft-backend-sampling' if c['draft_backend_sampling'] else '--no-spec-draft-backend-sampling']
        cmd += ['--spec-draft-device', c['device'], '--spec-draft-ngl', '0' if cpu else '99']
    for field, flag in [('cpu_mask', '--cpu-mask'), ('cpu_mask_batch', '--cpu-mask-batch')]:
        if c[field]:
            cmd += [flag, c[field]]
            if c['mtp_tokens']:
                cmd += [flag.replace('--cpu-', '--spec-draft-cpu-'), c[field]]
    if c['prefix_cache']:
        cmd += ['--ctx-checkpoints', '2', '--no-cache-idle-slots']
    if c['adaptive_mtp']:
        cmd += ['--spec-draft-adaptive']
    environment = {}
    if c['source_policy']:
        environment = {'LLAMA_SPEC_SOURCE_POLICY': '1', 'LLAMA_SPEC_SUFFIX': str(int(c['suffix_drafting']))}
        for field, key in [('policy_identity', 'LLAMA_SPEC_IDENTITY'), ('policy_input', 'LLAMA_SPEC_PROFILE_IN'),
                           ('policy_output', 'LLAMA_SPEC_PROFILE_OUT')]:
            if c[field]: environment[key] = c[field]
        if c['policy_frozen']: environment['LLAMA_SPEC_PROFILE_FROZEN'] = '1'
    plan = {'mode': 'dry-run', 'profile': c['id'], 'status': c['status'],
            'target_os': 'Linux', 'argv': cmd,
            'environment': environment,
            'transport_options': {'reuse_prefix': c['prefix_cache'], 'keep_alive': True},
            'endpoint': f"http://127.0.0.1:{c['port']}/v1",
            'request_defaults': request_defaults(c), 'required_model_sha256': MODEL_SHA256,
            'runtime_requirement': c['runtime_requirement'],
            'capabilities': {'native_xeno_graph': False, 'native_xeno_wire_protocol': False,
                             'native_xeno_snapshots': False, 'vision_validated': False,
                             'physical_kv_eviction': False, 'target_runtime_validated': False}}
    if c['source_policy']:
        try:
            from . import source_profiles as profiles
        except ImportError:
            import source_profiles as profiles
        plan['selector_settings'] = profiles.settings_for_plan(c, plan)
    return plan


def available_ram():
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot establish available RAM; refusing launch')


def verify_files(engine, model):
    exe, weights = Path(engine), Path(model)
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise ValueError('Provide an executable Linux Prism llama-server')
    with exe.open('rb') as handle:
        if handle.read(4) != b'\x7fELF':
            raise ValueError('Expected a Linux ELF engine, not a Windows runtime')
    digest = hashlib.sha256()
    with weights.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise ValueError('Expected GGUF model')
        handle.seek(0)
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError('Bonsai model digest mismatch')


def serve(plan, c):
    if platform.system() != 'Linux':
        raise RuntimeError('Launch is Linux-only; this host may inspect dry-run plans only')
    threshold = c['min_available_ram_gib'] * 2**30
    if available_ram() < threshold:
        raise RuntimeError('Insufficient available RAM before launch')
    verify_files(plan['argv'][0], plan['argv'][2])
    c = settings(c)
    if c['source_policy'] and c['policy_identity']:
        try:
            from . import source_profiles as profiles
        except ImportError:
            import source_profiles as profiles
        expected = profiles.identity(profiles.read(c['policy_identity']))
        if any(expected[k] != MODEL_SHA256 for k in
               ('weights_sha256', 'tokenizer_artifact_sha256', 'head_artifact_sha256')):
            raise ValueError('Selector model/tokenizer/head identity mismatch')
        profiles.check_runtime(expected, plan['argv'][0], profiles.settings_for_plan(c, plan))
        if c['policy_input']:
            profiles.validate(profiles.read(c['policy_input']), expected)
    # An occupied endpoint is another process, never something to attach to or kill.
    with socket.socket() as sock:
        if sock.connect_ex((c['host'], c['port'])) == 0:
            raise RuntimeError('Endpoint occupied; refusing duplicate server')
    env = os.environ.copy()
    # Host state must not silently replace pinned model/device/context choices.
    env = {k: v for k, v in env.items() if not k.startswith(('LLAMA_ARG_', 'LLAMA_SPEC_SOURCE_', 'LLAMA_SPEC_PROFILE_'))
           and k not in {'LLAMA_SPEC_IDENTITY', 'LLAMA_SPEC_SUFFIX'}}
    env.update(plan['environment'])
    child = None
    prior = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            prior[sig] = signal.signal(sig, interrupted)
        child = subprocess.Popen(plan['argv'], env=env, stdin=subprocess.DEVNULL)
        while child.poll() is None:
            if available_ram() < threshold:
                raise RuntimeError('Available RAM below reserve; stopping owned server')
            time.sleep(1)
        return child.returncode
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--engine', required=True, help='Owner-installed Linux Prism llama-server')
    parser.add_argument('--model', required=True)
    parser.add_argument('--start', action='store_true', help='Explicit Linux-only foreground launch')
    args = parser.parse_args(argv)
    c = load_profile(args.profile)
    plan = make_plan(c, args.engine, args.model)
    if not args.start:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        return serve(plan, c)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
