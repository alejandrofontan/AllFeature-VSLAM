# Information-based keyframe insertion

- **Date:** 2026-09-03
- **Kind:** note
- **Provenance:** moved verbatim from `CLAUDE.md` § *Information-Based Keyframe Insertion (2026-09-03)* on 2026-09-19 (documentation plan, `docs/plans/documentation_plan.md`). File/line references are as of the original date.


`Tracking::need_new_keyframe()` no longer uses the pixel-flow stationarity gate. Every tracked
frame is embedded (MegaLoc, ~4 ms) and `PlaceRecognition::keyframe_information(frame, local_keyframes_, centred)`
returns its **unexplained information** `v ∈ [0,1]` given Tracking's local map, computed by the new
read-only `placecell::MegaLocPlaceCell::unexplained_information(image, window, centred, &descriptor)`
on the same kernel/centring the information culler marginalises. Three bands, one shared key:

| band | condition | effect | key |
|---|---|---|---|
| redundant | `v < Tracking.KeyframeMinInformation` (default 0.05, unvalidated) | no keyframe (forced ones bypass) — replaces the flow gate | new |
| tracking health | `weak_tracking`, `low_overlap`, emergency | unchanged | — |
| novel | `v > tau` | keyframe even if tracking is fine | `LocalMapping.KeyframeCullingMaxUnexplained` (tau, also the culler's budget; Viewer slider "KF Max Unexplained") |

Schur identity: a keyframe inserted for novelty has unique information `v > tau` right after insertion (exact on the raw
kernel, approximate under centring), so it is never a cull candidate on its own account — insertion and culling
maintain the same "every view stays within tau of the alive keyframes" invariant from both ends. The frame's
embedding is cached in `Frame::global_descriptor`, copied into the new `KeyFrame::global_descriptor`, and
`PlaceRecognitionMegaLoc::compute(KeyFrame&)` stores it with `add()` instead of re-embedding (then releases image
and copy). `median_flow_from_last_frame()` is kept but only feeds the diagnostic line. `Tracking.LogKeyframeInformation: 1`
prints one line per tracked frame (info, tau, explainers, best explainer, flow, inliers, overlap, triggers, decision)
for threshold tuning; every insertion always logs a `need_new_keyframe: keyframe (<reasons>)` line with `info=`.
`vpr: none` yields no information → only the tracking triggers decide (no flow fallback, by decision).
Caveats: (1) the just-inserted reference keyframe is an explainer only once LocalMapping has processed it
(`process_new_keyframe` stores the descriptor), so for a frame or two after an insertion `v` is computed without
it — insertion is blocked while LocalMapping is busy anyway, except the emergency path; (2) the culler-side
degeneracy noted in `Thirdparty/placecell/CLAUDE.md` (centring set == alive set ⇒ singular `K_AA`) predates
this work and is not fixed here. Standalone brute-force test of the query: scratchpad `test_information.cpp`
(not in the repo); no SLAM run yet — thresholds await the user's logged runs.
