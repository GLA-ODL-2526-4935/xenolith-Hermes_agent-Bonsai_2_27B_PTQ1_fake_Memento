# Bonsai 27B PTQ1 persistent selector benchmark

Runtime: dedicated Prism-derived ff41412 source plus Intel v3/increment/source/profile/default deltas. CUDA build on the NVIDIA RTX 5080 Laptop GPU; these are not Intel laptop measurements.

All final conditions: temperature 0, low reasoning, reasoning budget 256, total output limit 2048, context8192, Q4_0 KV, one slot, no prompt cache. Selector cap4 and draft threshold0. The earlier unbounded pilot is saved separately and excluded.

24 authored Hermes-like workflows /72 frozen planning/tool/final states. Baseline covers all72; selector learns on54 calibration states, then reloads its saved prior frozen for18 held-out states. No real tools execute.

| Condition | Requests | Decode tokens/s | Screen pass | Token capped | Wall seconds |
|---|---:|---:|---:|---:|---:|
| No MTP - all states | 72 | 49.15 | 62 | 0 | 388.50 |
| No MTP - calibration states | 54 | 49.15 | 46 | 0 | 289.74 |
| Learning selector - calibration | 54 | 49.55 | 45 | 0 | 307.65 |
| No MTP - held-out states | 18 | 49.15 | 16 | 0 | 98.76 |
| Frozen selector - held-out | 18 | 49.26 | 15 | 0 | 129.46 |

Held-out aggregate decode throughput ratio: 1.002x. Same-question wall-time ratio (baseline/selector): 0.763x. Generated lengths may differ; time ratio is not a fixed-token speedup.

Learned observations (not acceptance percentages):

| Arm | Cycles | Committed tokens | Cycle tokens/s |
|---|---:|---:|---:|
| warm-off | 369 | 369 | 42.95 |
| mtp-1 | 41 | 65 | 36.38 |
| mtp-2 | 42 | 84 | 38.83 |
| mtp-3 | 41 | 98 | 40.99 |
| mtp-4 | 4313 | 11460 | 50.56 |
| suffix-1 | 43 | 51 | 33.90 |
| suffix-2 | 47 | 60 | 35.53 |
| suffix-3 | 81 | 115 | 40.64 |
| suffix-4 | 922 | 1398 | 47.61 |

Warm-off retains a synchronized MTP head, so it differs from true no-MTP. Arm timings include cycle source/draft/verify/replay cost; HTTP wall time also includes prefill. Aggregate decode TPS is total server predicted tokens / total server predicted time, including generated reasoning tokens.

Scores screen structure/tool arguments/required phrases only. They do not prove semantic accuracy. Exact final-answer strings can differ harmlessly; inspect raw answers, omissions, contradictions and capped cases. Selector optimization uses latency and committed tokens, never answer scores.

The learned JSON contains numeric counters and identity hashes only. Reuse through Xenolith external Prism on the matching runtime/hardware/settings; reset/recalibrate on Intel. The compiled default retains a process-wide bucket bank; the benchmark explicitly exports and reloads disk profiles.

Manual review: calibration introduced malformed JSON in failed-identical-call/plan and reasoning leakage in incomplete-range/plan; memory-staleness/final retained the planned/unverified qualification omitted by baseline. Held-out correction/plan regressed in formatting, while unknown/plan improved. Reasoning leakage was observed in both conditions, so this experiment does not establish its cause. Held-out provenance/final omitted the explicit instruction to clear counters although it retained the need to recalibrate. Denied-retry/final is a wording-screen false negative: its permitted public-read action is correct despite not repeating the word denied.

The held-out profile SHA256 was checked before/after and remained unchanged (84a324fd21b4e2d9d60e0f780986ff06b5f1d724ce8f0b4169dd3666532704ed). The saved 5899 cycles include one extra calibration completion retried after a transient Windows snapshot-read sharing failure; 54 unique calibration answers are scored. Launcher stdin/monitor handle errors were fixed and completed baseline/calibration records reused, rather than rerunning those phases. These are single sequential passes, without randomized repeated-run confidence intervals. The planning instructions may be ambiguous about treating the final response as a plan step; no prompt was changed between compared conditions.

Files: [per-question CSV](per-question-speed.csv), [full comparison JSON](reviewed-comparison.json), [saved prior](learned-profile.json), [live/final status](status.json). Raw baseline/learn/frozen JSONL retains content and reasoning.
