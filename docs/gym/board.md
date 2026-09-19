# Gym board

One row per gym sequence: the last measured numbers of the integration branch, when and at which
commit they were measured, and the entries that touched the sequence. Update the row with every
entry whose verdict is *kept*; a row older than the last kept change is flagged **re-measure**.

| Sequence | Config | Measured at | ATE RMSE (mm) | Losses | Keyframes | Median tracking (ms) | Wall (s) | Entries |
|---|---|---|---|---|---|---|---|---|
| `eth` / `table_3` | `exp_gym_mono` | 2026-08-16, P5 confirmation build (pre-issue-#15/#31/#32 cleanup, pre-MegaLoc, features `orb32+aliked128`) — **re-measure** | 4.8–6.5 | 0 | 66–73 | tracking 37 (profiling block; `median tracking time` not recorded then) | 80 | [vanilla](entries/2026-08-16_vanilla-baseline.md) · [m1-mimalloc](entries/2026-08-16_m1-mimalloc.md) · [p3-g2o-openmp](entries/2026-08-16_p3-g2o-openmp.md) · [p2-bruteforce-matcher](entries/2026-08-16_p2-bruteforce-matcher.md) · [l1-singletons](entries/2026-08-16_l1-singletons.md) · [m2-copies](entries/2026-08-16_m2-copies.md) · [g2-lightglue-cache](entries/2026-08-16_g2-lightglue-cache.md) · [tr-overlap-batch](entries/2026-08-16_tr-overlap-batch.md) · [p5-parallel-fuse](entries/2026-08-16_p5-parallel-fuse.md) · [aliked-only](entries/2026-08-16_aliked-only.md) |

## Benchmark set

Issue #18's 100 sequences from 100 datasets; run at milestones only, as the regression check of
kept gym changes. No milestone run recorded yet.

## Candidates for the gym set

- NSAVP `R0_FA0` (driving, stops, 19103 frames, RGB-D with `fastfoundationstereo` depth): the
  sequence of the 2026-08-12/13 tracking-loss investigations (`docs/notes/`). Long (an hour per
  run); a cropped range (`rgb_idx`) around the stop sections would make it a daily sequence.
