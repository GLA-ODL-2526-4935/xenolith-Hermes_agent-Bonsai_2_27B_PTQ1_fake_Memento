"""Offline backend contracts. Never spawn inference or open real weights."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import platform
import unittest
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_backend', ENGINE / 'backends/prism.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BackendContract(unittest.TestCase):
    def setUp(self):
        self.c = b.load_profile(ENGINE / 'profiles/bonsai27-linux-cpu.json')

    def test_dry_run_does_not_need_model_or_engine(self):
        args = ['--profile', str(ENGINE / 'profiles/bonsai27-linux-cpu.json'),
                '--engine', '/not-installed/llama-server', '--model', '/not-present/' + b.MODEL]
        with patch.object(b.subprocess, 'Popen', side_effect=AssertionError('No engine allowed')) as spawn:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(b.main(args), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['mode'], 'dry-run')
            self.assertFalse(plan['capabilities']['native_xeno_snapshots'])
            spawn.assert_not_called()

    def test_reject_incompatible_model(self):
        with self.assertRaises(ValueError):
            b.make_plan(self.c, '/engine', '/models/gemma.gguf')

    def test_loopback_only(self):
        self.c['host'] = '0.0.0.0'
        with self.assertRaises(ValueError):
            b.validate(self.c)

    def test_no_silent_gpu_offload_cpu(self):
        cmd = b.make_plan(self.c, '/engine', '/models/' + b.MODEL)['argv']
        self.assertEqual(cmd[cmd.index('-ngl') + 1], '0')
        self.assertEqual(cmd[cmd.index('--device') + 1], 'none')
        self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'none')

    def test_explicit_vulkan_and_conservative_mtp(self):
        c = b.load_profile(ENGINE / 'profiles/bonsai27-linux-vulkan.json')
        cmd = b.make_plan(c, '/engine', '/models/' + b.MODEL)['argv']
        self.assertEqual(cmd[cmd.index('--device') + 1], 'Vulkan0')
        self.assertEqual(cmd[cmd.index('--spec-draft-n-max') + 1], '1')

    def test_cpu_tuning_flags_are_separate_and_never_offload(self):
        c = {**self.c, 'threads': 4, 'threads_batch': 8, 'cpu_mask': '0f',
             'cpu_mask_batch': 'ff', 'poll': 0, 'poll_batch': 1, 'mtp_tokens': 3,
             'draft_backend_sampling': False}
        plan = b.make_plan(c, '/engine', '/models/' + b.MODEL)
        cmd = plan['argv']
        for flag, value in [('-t', '4'), ('-tb', '8'), ('--cpu-mask', '0f'),
                            ('--cpu-mask-batch', 'ff'), ('--poll', '0'),
                            ('--poll-batch', '1'), ('--spec-draft-threads', '4'),
                            ('--spec-draft-threads-batch', '8')]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--no-spec-draft-backend-sampling', cmd)
        self.assertEqual(plan['environment'], {})
        self.assertEqual(cmd[cmd.index('-ngl') + 1], '0')

    def test_prefix_reuse_is_explicit_and_bounded(self):
        off = b.make_plan(self.c, '/engine', '/models/' + b.MODEL)
        self.assertFalse(off['request_defaults']['cache_prompt'])
        c = {**self.c, 'prefix_cache': True, 'cache_ram_mib': 128}
        on = b.make_plan(c, '/engine', '/models/' + b.MODEL)
        self.assertTrue(on['request_defaults']['cache_prompt'])
        self.assertEqual(on['request_defaults']['id_slot'], 0)
        self.assertIn('--no-cache-idle-slots', on['argv'])
        self.assertEqual(on['argv'][on['argv'].index('--cache-ram') + 1], '128')

    def test_reject_invalid_cpu_policy(self):
        for extra in [{'threads_batch': 0}, {'cpu_mask': '0'}, {'cpu_mask': '0x1'},
                      {'poll': True}, {'poll': 100}, {'cache_ram_mib': 1024},
                      {'cache_ram_mib': 128}, {'mtp_tokens': 4, 'ubatch': 4},
                      {'adaptive_mtp': True}, {'prefix_cache': 'true'}]:
            with self.assertRaises(ValueError, msg=str(extra)):
                b.validate({**self.c, **extra})

    def test_adaptive_candidate_requires_patched_runtime(self):
        c = b.load_profile(ENGINE / 'profiles/bonsai27-linux-cpu-candidate.json')
        cmd = b.make_plan(c, '/engine', '/models/' + b.MODEL)['argv']
        self.assertIn('--spec-draft-adaptive', cmd)
        with self.assertRaises(ValueError):
            b.validate({**c, 'runtime_requirement': 'unpatched Prism'})
        with self.assertRaises(ValueError):
            b.validate({**c, 'mtp_probability': 0.3})

    def test_reasoning_low_and_bounded_output(self):
        p = b.request_defaults(self.c)
        self.assertEqual(p['reasoning_effort'], 'low')
        self.assertTrue(p['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(p['max_tokens'], 4096)

    def test_greedy_server_and_client_defaults_agree(self):
        for profile in ['bonsai27-linux-cpu.json', 'bonsai27-linux-vulkan.json']:
            c = b.load_profile(ENGINE / 'profiles' / profile)
            plan = b.make_plan(c, '/engine', '/models/' + b.MODEL)
            cmd, request = plan['argv'], plan['request_defaults']
            for flag, key in [('--temp', 'temperature'), ('--top-k', 'top_k'),
                              ('--top-p', 'top_p'), ('--min-p', 'min_p'),
                              ('--repeat-penalty', 'repeat_penalty'), ('--seed', 'seed')]:
                self.assertEqual(float(cmd[cmd.index(flag) + 1]), request[key])
            self.assertEqual(cmd[cmd.index('--samplers') + 1], ';'.join(request['samplers']))
            self.assertEqual(request['samplers'], ['top_k', 'temperature'])

    def test_refuse_coerced_or_oversized_context(self):
        for value in [True, '8192', 999999]:
            c = copy.deepcopy(self.c)
            c['context'] = value
            with self.assertRaises(ValueError):
                b.validate(c)

    def test_refuse_output_exhausting_context(self):
        self.c['max_output_tokens'] = self.c['context']
        with self.assertRaises(ValueError):
            b.validate(self.c)

    def test_refuse_low_memory_reserve(self):
        self.c['min_available_ram_gib'] = 1
        with self.assertRaises(ValueError):
            b.validate(self.c)

    def test_reject_unknown_and_duplicate_profile_fields(self):
        for content in [json.dumps({**self.c, 'mpt_tokens': 3}), '{"id":1,"id":2}']:
            with patch.object(Path, 'read_text', return_value=content):
                with self.assertRaises(ValueError):
                    b.load_profile('unused-profile-path')

    @unittest.skipIf(platform.system() == 'Linux', 'Non-Linux launch refusal only')
    def test_wrong_os_refuses_before_any_engine_or_model_access(self):
        plan = b.make_plan(self.c, '/not-present', '/not-present/' + b.MODEL)
        with patch.object(b, 'verify_files', side_effect=AssertionError('Must not open files')):
            with self.assertRaisesRegex(RuntimeError, 'Linux-only'):
                b.serve(plan, self.c)


if __name__ == '__main__':
    unittest.main()
