#!/usr/bin/env python3
# AI Identity Incorported <james@aiii.id>
"""Offline, fixed-payload expert-cache trace analysis. Never opens model shards."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, pathlib, subprocess, sys
from dataclasses import asdict, dataclass

GIB = 1024 ** 3
SPECS = {
    'ring-1031': {'model': 'Ring-mini-2.0', 'layers': tuple(range(1, 20)), 'experts': 256,
                  'payload_bytes': 6 * 1024 * 1024},
    'ling-195': {'model': 'Ling-3.0-tiny', 'layers': tuple(range(1, 24)), 'experts': 128,
                 'payload_bytes': int(4.5 * 1024 * 1024)},
    'ling-358': {'model': 'Ling-3.0-tiny', 'layers': tuple(range(1, 24)), 'experts': 128,
                 'payload_bytes': int(4.5 * 1024 * 1024)},
}

@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    resident_bytes: int
    streamed_payload_bytes: int


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_native_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'execution receipt {name} must be a non-negative integer')
    return value


def validate_receipt(receipt: dict, spec: dict) -> int:
    if receipt.get('model') != spec['model']:
        raise ValueError(f'execution receipt model mismatch: {receipt.get("model")!r}')
    for name in ('tokens', 'cache_budget_bytes', 'cache_hits', 'cache_misses',
                 'cache_evictions', 'cache_resident_bytes'):
        require_native_int(receipt.get(name), name)
    if receipt['cache_budget_bytes'] != GIB:
        raise ValueError('execution receipt cache_budget_bytes must equal 1 GiB')
    return receipt['tokens']


def parse_routes(path: pathlib.Path, *, tokens: int, layers: tuple[int, ...], experts: int) -> list[tuple[int, int]]:
    """Return accesses in exact executor order, rejecting incomplete/corrupt traces."""
    if tokens <= 0:
        raise ValueError('execution receipt tokens must be positive')
    accesses: list[tuple[int, int]] = []
    rows: set[tuple[int, int]] = set()
    last_position = -1
    last_layer = -1
    valid_layers = set(layers)
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        f = raw.split('\t')
        if len(f) != 10:
            raise ValueError(f'{path}:{lineno}: expected position, layer, and 8 routes')
        try:
            position, layer = int(f[0]), int(f[1])
        except ValueError as e:
            raise ValueError(f'{path}:{lineno}: non-integer position/layer') from e
        if not (0 <= position < tokens) or layer not in valid_layers:
            raise ValueError(f'{path}:{lineno}: model-specific position/layer bound violation')
        if position < last_position or (position == last_position and layer <= last_layer):
            raise ValueError(f'{path}:{lineno}: routes are not in executor order')
        if (position, layer) in rows:
            raise ValueError(f'{path}:{lineno}: duplicate position/layer row')
        selected: set[int] = set()
        for route in f[2:]:
            if route.count(':') != 1:
                raise ValueError(f'{path}:{lineno}: malformed route {route!r}')
            expert_s, weight_s = route.split(':')
            try:
                expert, weight = int(expert_s), float(weight_s)
            except ValueError as e:
                raise ValueError(f'{path}:{lineno}: malformed route {route!r}') from e
            if not (0 <= expert < experts) or expert in selected or not math.isfinite(weight):
                raise ValueError(f'{path}:{lineno}: invalid expert selection or weight')
            selected.add(expert)
            accesses.append((layer, expert))
        rows.add((position, layer))
        last_position, last_layer = position, layer
    expected = {(position, layer) for position in range(tokens) for layer in layers}
    if rows != expected:
        missing, extra = expected - rows, rows - expected
        raise ValueError(f'{path}: missing/extra route rows missing={len(missing)} extra={len(extra)}')
    return accesses


def lru(accesses: list[tuple[int, int]], capacity: int, payload_bytes: int) -> CacheStats:
    if capacity <= 0:
        raise ValueError('capacity must be positive')
    live: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
    hits = misses = evictions = 0
    for key in accesses:
        if key in live:
            hits += 1
            live.move_to_end(key)
            continue
        misses += 1
        while len(live) >= capacity:
            live.popitem(last=False)
            evictions += 1
        live[key] = None
    return CacheStats(hits, misses, evictions, len(live), len(live) * payload_bytes,
                      misses * payload_bytes)


def belady(accesses: list[tuple[int, int]], capacity: int, payload_bytes: int) -> CacheStats:
    """Optimal replacement for equal-size payloads, using the next-use sequence."""
    if capacity <= 0:
        raise ValueError('capacity must be positive')
    next_use = [len(accesses)] * len(accesses)
    later: dict[tuple[int, int], int] = {}
    for i in range(len(accesses) - 1, -1, -1):
        key = accesses[i]
        next_use[i] = later.get(key, len(accesses))
        later[key] = i
    live: dict[tuple[int, int], int] = {}
    hits = misses = evictions = 0
    for i, key in enumerate(accesses):
        if key in live:
            hits += 1
            live[key] = next_use[i]
            continue
        misses += 1
        if len(live) >= capacity:
            victim = max(live, key=lambda k: live[k])
            del live[victim]
            evictions += 1
        live[key] = next_use[i]
    return CacheStats(hits, misses, evictions, len(live), len(live) * payload_bytes,
                      misses * payload_bytes)


def check_current(sim: CacheStats, receipt: dict, payload_bytes: int) -> dict:
    expected = {k: receipt[k] for k in ('cache_hits', 'cache_misses', 'cache_evictions', 'cache_resident_bytes')}
    actual = {'cache_hits': sim.hits, 'cache_misses': sim.misses,
              'cache_evictions': sim.evictions, 'cache_resident_bytes': sim.resident_bytes}
    return {'expected': expected, 'actual': actual, 'exact_match': expected == actual,
            'payload_bytes': payload_bytes}


def summarize(name: str, native: pathlib.Path) -> dict:
    spec = SPECS[name]
    routes, execution = native / 'routes.tsv', native / 'execution.json'
    receipt = json.loads(execution.read_text())
    tokens = validate_receipt(receipt, spec)
    accesses = parse_routes(routes, tokens=tokens, layers=spec['layers'], experts=spec['experts'])
    payload_bytes = spec['payload_bytes']
    current_capacity = GIB // payload_bytes
    current = lru(accesses, current_capacity, payload_bytes)
    calibration = check_current(current, receipt, payload_bytes)
    unique_total = len(set(accesses))
    unique_by_layer = {str(layer): len({expert for candidate, expert in accesses if candidate == layer})
                       for layer in spec['layers']}
    result = {
        'case': name, 'native_dir': str(native), 'route_sha256': sha256(routes),
        'execution_sha256': sha256(execution), 'receipt_model': receipt['model'],
        'receipt_tokens': tokens, 'route_lines': len(accesses) // 8, 'accesses': len(accesses),
        'unique_experts_total': unique_total, 'unique_experts_by_layer': unique_by_layer,
        'payload_bytes': payload_bytes, 'current_budget_bytes': GIB,
        'capacities_entries': {'1GiB': current_capacity, '2GiB': (2 * GIB) // payload_bytes},
        'current_lru_calibration': calibration, 'counterfactual_admissible': calibration['exact_match'],
        'current_lru': asdict(current),
        'interpretation': ('streamed_payload_bytes is a model-implied payload volume: cache misses multiplied by fixed BF16 expert bytes. It is not measured transport, I/O, or latency.'),
    }
    if calibration['exact_match']:
        result['counterfactual_status'] = 'admitted_after_exact_1GiB_calibration'
        result['lru'] = {label: asdict(lru(accesses, cap, payload_bytes))
                         for label, cap in result['capacities_entries'].items()}
        result['belady_lower_bound'] = {label: asdict(belady(accesses, cap, payload_bytes))
                                        for label, cap in result['capacities_entries'].items()}
    else:
        result['counterfactual_status'] = 'suppressed_due_to_failed_1GiB_calibration'
        result['lru'] = None
        result['belady_lower_bound'] = None
    return result


def source_pins(root: pathlib.Path) -> dict:
    files = ['src/ring/expert_cache.cpp', 'src/ring/engine.cu', 'src/ling/engine.cu',
             'tools/ling_ring_bf16/CMakeLists.txt', 'tools/ling_ring_bf16/ling/CMakeLists.txt']
    return {'source_root': str(root),
            'git_head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'files_sha256': {name: sha256(root / name) for name in files}}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--source-root', type=pathlib.Path, required=True)
    p.add_argument('--ring-native', type=pathlib.Path, required=True)
    p.add_argument('--ling-195-native', type=pathlib.Path, required=True)
    p.add_argument('--ling-358-native', type=pathlib.Path, required=True)
    args = p.parse_args(argv)
    cases = [summarize('ring-1031', args.ring_native), summarize('ling-195', args.ling_195_native),
             summarize('ling-358', args.ling_358_native)]
    out = {'schema': 'strata-expert-trace-analysis-v2',
           'scope': 'Offline simulation only. No model, tensor payload, GPU, or service operation.',
           'policy': 'Global equal-payload LRU in exact routes.tsv order; Belady is an offline clairvoyant lower bound, not an executable policy.',
           'source_pins': source_pins(args.source_root), 'analyzer_sha256': sha256(pathlib.Path(__file__)),
           'cases': cases}
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    return 0 if all(x['counterfactual_admissible'] for x in cases) else 2

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
