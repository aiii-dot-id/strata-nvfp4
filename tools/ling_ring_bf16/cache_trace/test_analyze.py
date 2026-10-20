# AI Identity Incorported <james@aiii.id>
import json, pathlib, tempfile, unittest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analyze import GIB, SPECS, belady, lru, parse_routes, summarize, validate_receipt

class CacheTests(unittest.TestCase):
    def test_hand_lru(self):
        s = lru([('L', 0), ('L', 1), ('L', 0), ('L', 2)], 2, 10)
        self.assertEqual((s.hits, s.misses, s.evictions, s.entries, s.resident_bytes, s.streamed_payload_bytes), (1, 3, 1, 2, 20, 30))

    def test_hand_belady_beats_lru(self):
        trace = [('L', x) for x in 'ABCABC']
        self.assertEqual(lru(trace, 2, 10).misses, 6)
        self.assertEqual(belady(trace, 2, 10).misses, 4)

    def test_invalid_inf_duplicate_and_model_bounds_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'routes.tsv'
            base = ['0', '1'] + [f'{i}:0.1' for i in range(8)]
            p.write_text('\t'.join(base[:-1] + ['7:inf']) + '\n')
            with self.assertRaises(ValueError): parse_routes(p, tokens=1, layers=(1,), experts=8)
            p.write_text('\t'.join(['0', '2'] + [f'{i}:0.1' for i in range(8)]) + '\n')
            with self.assertRaises(ValueError): parse_routes(p, tokens=1, layers=(1,), experts=8)
            p.write_text('\t'.join(['0', '1'] + [f'{i}:0.1' for i in range(7)] + ['8:0.1']) + '\n')
            with self.assertRaises(ValueError): parse_routes(p, tokens=1, layers=(1,), experts=8)

    def test_missing_position_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'routes.tsv'
            p.write_text('\t'.join(['0', '1'] + [f'{i}:0.1' for i in range(8)]) + '\n')
            with self.assertRaises(ValueError): parse_routes(p, tokens=2, layers=(1,), experts=8)

    def test_receipt_type_and_budget_rejected(self):
        r = {'model': 'Ring-mini-2.0', 'tokens': 1, 'cache_budget_bytes': GIB,
             'cache_hits': 0, 'cache_misses': 8, 'cache_evictions': 0, 'cache_resident_bytes': 8}
        validate_receipt(r, SPECS['ring-1031'])
        r['cache_hits'] = True
        with self.assertRaises(ValueError): validate_receipt(r, SPECS['ring-1031'])
        r['cache_hits'] = 0; r['cache_budget_bytes'] = GIB - 1
        with self.assertRaises(ValueError): validate_receipt(r, SPECS['ring-1031'])

    def test_failed_calibration_suppresses_counterfactuals(self):
        with tempfile.TemporaryDirectory() as d:
            n = pathlib.Path(d)
            rows = ['\t'.join(['0', str(layer)] + [f'{i}:0.1' for i in range(8)]) for layer in range(1, 20)]
            (n / 'routes.tsv').write_text('\n'.join(rows) + '\n')
            (n / 'execution.json').write_text(json.dumps({'model':'Ring-mini-2.0','tokens':1,'cache_budget_bytes':GIB,'cache_hits':1,'cache_misses':151,'cache_evictions':0,'cache_resident_bytes':956301312}))
            x = summarize('ring-1031', n)
            self.assertFalse(x['counterfactual_admissible'])
            self.assertIsNone(x['lru']); self.assertIsNone(x['belady_lower_bound'])

if __name__ == '__main__':
    unittest.main()
