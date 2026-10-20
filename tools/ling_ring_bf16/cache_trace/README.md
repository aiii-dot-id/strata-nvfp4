# Offline expert-cache trace analysis

**Attribution:** AI Identity Incorported  
**Scope:** retained `routes.tsv` and `execution.json` only. No model, tensor payload, GPU, service, or source-tree mutation occurred.

`analyze.py` replays expert selections in their recorded position/layer/expert order under the current global equal-payload LRU policy. It first requires exact agreement with the native one-GiB receipt. Only calibrated cases receive two-GiB and offline Belady comparisons.

## Calibration

All three current one-GiB simulations exactly match the native receipt's hits, misses, evictions, and resident bytes:

| Trace | accesses | unique `(layer, expert)` | payload | 1 GiB entries | native/simulated hits | misses | evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ring 1031 | 156,712 | 3,375 | 6 MiB | 170 | 45,588 | 111,124 | 110,954 |
| Ling 195 | 35,880 | 2,448 | 4.5 MiB | 227 | 12,242 | 23,638 | 23,411 |
| Ling 358 | 65,872 | 2,470 | 4.5 MiB | 227 | 25,338 | 40,534 | 40,307 |

Per-layer unique counts and input hashes are in `RESULTS.json`.

## Counterfactual cache statistics

| Trace | Policy | capacity | misses | hits | evictions | implied payload-on-miss bytes |
|---|---|---:|---:|---:|---:|---:|
| Ring 1031 | LRU | 1 GiB / 170 | 111,124 | 45,588 | 110,954 | 699,131,756,544 |
| Ring 1031 | LRU | 2 GiB / 341 | 92,049 | 64,663 | 91,708 | 579,122,233,344 |
| Ring 1031 | Belady lower bound | 1 GiB / 170 | 73,318 | 83,394 | 73,148 | 461,276,971,008 |
| Ring 1031 | Belady lower bound | 2 GiB / 341 | 50,961 | 105,751 | 50,620 | 320,618,889,216 |
| Ling 195 | LRU | 1 GiB / 227 | 23,638 | 12,242 | 23,411 | 111,538,077,696 |
| Ling 195 | LRU | 2 GiB / 455 | 18,610 | 17,270 | 18,155 | 87,812,997,120 |
| Ling 195 | Belady lower bound | 1 GiB / 227 | 15,510 | 20,370 | 15,283 | 73,185,361,920 |
| Ling 195 | Belady lower bound | 2 GiB / 455 | 10,428 | 25,452 | 9,973 | 49,205,477,376 |
| Ling 358 | LRU | 1 GiB / 227 | 40,534 | 25,338 | 40,307 | 191,263,408,128 |
| Ling 358 | LRU | 2 GiB / 455 | 30,417 | 35,455 | 29,962 | 143,525,412,864 |
| Ling 358 | Belady lower bound | 1 GiB / 227 | 26,117 | 39,755 | 25,890 | 123,235,467,264 |
| Ling 358 | Belady lower bound | 2 GiB / 455 | 15,734 | 50,138 | 15,279 | 74,242,326,528 |

“Implied payload-on-miss bytes” is misses multiplied by the fixed original-BF16 expert payload size. It is not a measured H2D transfer total, disk I/O, host residency, CUDA time, or wall-clock prediction. Belady uses future knowledge and is a lower bound, not an implementation proposal.

## Reproduction and controls

```text
python3 -m pytest -q tools/ling_ring_bf16/cache_trace
python3 tools/ling_ring_bf16/cache_trace/analyze.py --output RESULTS.json --source-root /work/AI/src/strata-ling-ring-lmhead-20261020 --ring-native .../ring-1031/native --ling-195-native .../ling-195/native --ling-358-native .../ling-358/native
```

Controls include hand-verifiable LRU counters, a trace on which Belady strictly beats LRU, and malformed/duplicate-route rejection. `RESULTS.json` pins every input trace/receipt, the analyzer, and the source files that define the shared cache semantics.

The parser rejects nonfinite weights, wrong model/layer/expert geometry, missing routes, noninteger counters, and budgets other than the calibrated one GiB. Failed counter calibration suppresses both counterfactual tables and exits nonzero.
