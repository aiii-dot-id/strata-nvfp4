import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import native


class NativeSamplingTests(unittest.TestCase):
    def _receipt_and_logits(self, root):
        logits = np.zeros((1, 157184), dtype='<f4')
        logits[0, 0] = 2.0
        logits[0, 1] = 1.0
        path = Path(root) / 'logits.f32'
        logits.tofile(path)
        temperature = 1.0
        total = math.exp(0.0) + math.exp(-1.0) + (157184 - 2) * math.exp(-2.0)
        uniform = 0.000001
        token = 0
        logprob = -math.log(total)
        generation = {
            'schema': 'strata-ling-generation-v1',
            'prompt_token_ids': [1],
            'generated_token_ids': [token],
            'eos_token_ids': [99],
            'max_new_tokens': 1,
            'finish_reason': 'length',
            'sampling': {
                'method': 'categorical_full_vocab',
                'temperature': temperature,
                'seed': 7,
                'algorithm': 'mt19937_64_double_max_shift',
                'sampled_logprobs': [logprob],
                'uniform_draws': [uniform],
            },
        }
        return generation, path

    def test_full_vocab_receipt_audit_accepts_stable_underflow_tail(self):
        with tempfile.TemporaryDirectory() as d:
            generation, logits = self._receipt_and_logits(d)
            native._validate_generation(
                generation, [1], 1, [99], [1], 1.0, 7, logits
            )

    def test_sampling_receipt_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            generation, logits = self._receipt_and_logits(d)
            for key, value in (
                ('uniform_draws', [1.0]),
                ('sampled_logprobs', [-1.0]),
            ):
                bad = json.loads(json.dumps(generation))
                bad['sampling'][key] = value
                with self.assertRaises(RuntimeError):
                    native._validate_generation(
                        bad, [1], 1, [99], [1], 1.0, 7, logits
                    )

    def test_default_generation_rejects_unexpected_sampling_object(self):
        with tempfile.TemporaryDirectory() as d:
            generation, _ = self._receipt_and_logits(d)
            with self.assertRaises(RuntimeError):
                native._validate_generation(generation, [1], 1, [99], [1])

    def test_sampling_arguments_are_strictly_paired_and_typed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=1.0)
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=True,
                           sampling_seed=1, generate=1, eos=[99])
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=1.0,
                           sampling_seed=True, generate=1, eos=[99])
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=0.0,
                           sampling_seed=1, generate=1, eos=[99])
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=1.0,
                           sampling_seed=1 << 64, generate=1, eos=[99])
            with self.assertRaises(ValueError):
                native.run('ling', [1], out, sampling_temperature=1.0,
                           sampling_seed=1, generate=0)


if __name__ == '__main__':
    unittest.main()
