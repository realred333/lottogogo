# Pool Exclusion Experiment Notes

Date: 2026-08-01

This document summarizes the local LottoGoGo exclusion-pool experiment. The original
`recommend.py` and `backtest.py` were intentionally left unchanged. New work was
added in separate experimental files.

## Goal

The experiment changed the evaluation target from "which final recommended games
hit" to "how many actual winning numbers survive after exclusions."

Primary metric:

```text
For each target round:
1. Train only on rounds before the target round.
2. Build the available number pool.
3. Compare the target round's six winning numbers against the pool.
4. Count how many of the six survived.
```

Example output format to use going forward:

```text
pool_size=30
6 survived: n rounds
5 survived: n rounds
4 survived: n rounds
3 survived: n rounds
2 survived: n rounds
1 survived : n rounds
0 survived : n rounds
average survived: x.xxx
worst rounds: ...
```

## Files Added

### `recommend_pool.py`

Experimental pool-based recommender.

Important behavior:

```text
hard_exclude is removed before sampling.
risk_exclude is selected from the remaining numbers to fit pool_size.
Sampling is broad/uniform from the final pool.
Original recommend.py is not modified.
```

Current hard exclusions:

```text
{2, 5, 8, 9, 22, 32, 39}
```

Pool construction:

```text
1. Start with 1..45.
2. Remove hard_exclude numbers.
3. 38 numbers remain.
4. If pool_size=30, remove 8 more numbers.
5. Those 8 automatic removals are risk_exclude.
```

Current risk score formula:

```text
risk_score =
0.40 * long_gap
+ 0.25 * silence_low
+ 0.25 * rare_pair
+ 0.10 * overheat
```

Risk components:

```text
long_gap:
  - average reappearance gap
  - count of gaps >= 30 rounds
  - max gap
  - 90th percentile gap

silence_low:
  - current gap since last appearance
  - low recent-50 frequency

rare_pair:
  - pair count <= 10
  - pair count <= 12
  - low average pair count

overheat:
  - recent-20 frequency
  - recent-5 frequency
```

The `avg_gap` component was temporarily disabled during testing and then restored.
Current state applies `avg_gap` normally:

```text
long_gap =
0.35 * avg_gap
+ 0.25 * gap_ge30
+ 0.20 * max_gap
+ 0.20 * p90_gap
```

### `backtest_pool.py`

Backtest wrapper for `recommend_pool.py`.

Important behavior:

```text
--target-round means the actual target round.
Training data is all rounds strictly before the target round.
Original backtest.py is not modified.
```

## Current Latest Data

Local `history.csv` latest round:

```text
1234
```

Full-data next-round pool example using `pool_size=30`:

```text
hard_exclude = [2, 5, 8, 9, 22, 32, 39]
risk_exclude = [6, 10, 11, 23, 25, 28, 29, 41]
pool = [1, 3, 4, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 26, 27, 30, 31, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45]
```

## Survival Backtest Results

Recent 100 rounds, target range `1135..1234`, `pool_size=30`:

```text
6 survived:  9 rounds
5 survived: 28 rounds
4 survived: 30 rounds
3 survived: 20 rounds
2 survived: 10 rounds
1 survived :  3 rounds
0 survived :  0 rounds

average survived: 3.970
```

Summary:

```text
5+ survived: 37 / 100
4+ survived: 67 / 100
3 or fewer survived: 33 / 100
2 or fewer survived: 13 / 100
```

Rounds where all six survived:

```text
1135: [1, 6, 13, 19, 21, 33]
1141: [7, 11, 12, 21, 26, 35]
1144: [3, 4, 12, 15, 26, 34]
1155: [10, 16, 19, 27, 37, 38]
1179: [3, 16, 18, 24, 40, 44]
1194: [3, 13, 15, 24, 33, 37]
1195: [3, 15, 27, 33, 34, 36]
1215: [13, 15, 19, 21, 44, 45]
1231: [4, 13, 14, 18, 31, 38]
```

Worst rounds, two or fewer survived:

```text
1142: survived=1, actual=[2, 8, 28, 30, 37, 41]
1162: survived=2, actual=[20, 21, 22, 25, 28, 29]
1166: survived=2, actual=[14, 23, 25, 27, 29, 42]
1167: survived=2, actual=[8, 23, 31, 35, 39, 40]
1185: survived=1, actual=[6, 17, 22, 28, 29, 32]
1186: survived=2, actual=[2, 8, 13, 16, 23, 28]
1193: survived=2, actual=[6, 9, 16, 19, 24, 28]
1204: survived=2, actual=[8, 16, 28, 30, 31, 44]
1212: survived=2, actual=[5, 8, 25, 31, 41, 44]
1220: survived=1, actual=[2, 22, 25, 28, 34, 43]
1222: survived=2, actual=[4, 11, 17, 22, 32, 41]
1225: survived=2, actual=[8, 9, 19, 25, 41, 42]
1230: survived=2, actual=[3, 8, 9, 22, 28, 42]
```

## Exclusion Cause Analysis

For rounds with 5 or 4 surviving numbers:

```text
5 survived: 28 rounds
4 survived: 30 rounds
total: 58 rounds
```

These rounds had 88 actual winning numbers removed:

```text
risk_exclude: 46 removed winning numbers
hard_exclude: 42 removed winning numbers
```

Breakdown:

```text
5-survived rounds:
  risk_exclude: 16
  hard_exclude: 12

4-survived rounds:
  risk_exclude: 30
  hard_exclude: 30
```

Most frequently removed actual winning numbers in these 58 rounds:

```text
9  : 10 hard
29 :  8 risk
30 :  7 risk
32 :  7 hard
23 :  7 risk
39 :  6 hard
5  :  5 hard
22 :  5 hard
6  :  5 risk
8  :  5 hard
2  :  4 hard
41 :  4 risk
28 :  3 risk
25 :  3 risk
42 :  2 risk
16 :  2 risk
24 :  1 risk
11 :  1 risk
44 :  1 risk
10 :  1 risk
43 :  1 risk
```

Risk component cause analysis for the 46 risk-removed winning numbers:

```text
major component:
long_gap    : 42
silence_low :  3
rare_pair   :  1
overheat    :  0
```

Top sub-component:

```text
avg_gap        : 27
rare_pair_le10 : 10
current_gap    :  5
gap_ge30       :  2
recent50_low   :  2
```

Interpretation:

```text
The current risk_exclude misses are mostly caused by long_gap.
Inside long_gap, avg_gap is the biggest contributor.
However, disabling avg_gap alone did not improve survival because pool_size=30 still forces 8 automatic removals.
When avg_gap was disabled, rare_pair_le10 became the next main removal driver.
```

## Temporary avg_gap Test

`avg_gap` was temporarily changed from:

```text
0.35 * avg_gap
```

to:

```text
0.00 * avg_gap
```

Result, recent 100 rounds, `pool_size=30`:

```text
6 survived:  9
5 survived: 28
4 survived: 29
3 survived: 21
2 survived: 10
1 survived :  3
0 survived :  0

average survived: 3.960
```

This was slightly worse than the original `avg_gap` enabled result:

```text
average survived with avg_gap enabled : 3.970
average survived with avg_gap disabled: 3.960
```

`avg_gap` was restored after this test.

## Working Interpretation

Current state:

```text
hard_exclude is a fixed manual exclusion list.
risk_exclude is an automatic ranking-based exclusion list.
pool_size=30 means hard exclusions plus 8 additional automatic risk exclusions.
```

The current pool-reduction strategy is not yet strong enough to justify aggressive
shrinking below 30. It often removes real winning numbers, especially through
long-gap style scoring.

Practical next direction:

```text
1. Keep the original recommend.py/backtest.py untouched.
2. Keep recommend_pool.py/backtest_pool.py as experimental files.
3. Evaluate future changes only through survival distribution first.
4. Do not judge by final generated games until pool survival improves.
5. Tune risk_exclude by comparing survival distribution before/after every change.
```

Suggested next experiments:

```text
1. Compare pool_size 35, 32, 30, 28, 25 using the same survival table.
2. Reduce forced removals before changing many score weights.
3. Test removing or lowering rare_pair_le10 after avg_gap, but only through survival distribution.
4. Consider a "protect list" for numbers that risk_exclude repeatedly kills but appear often in backtests, such as 23, 29, 30.
```
