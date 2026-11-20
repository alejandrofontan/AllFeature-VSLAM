# Review findings

The working list for AllFeature-VSLAM: every open GitHub issue of `alejandrofontan/AllFeature-VSLAM`,
clustered and ordered for solving. `review-lenses.md` defines the lenses and the labels.

**Rules**

- GitHub is the source of the issues; this file is the order of work. `#` is the GitHub issue number.
- A row leaves this file when its issue is closed on GitHub (with the commit in the closing comment).
- A new finding gets a GitHub issue first, then a row here.
- In conversation, show only the rows of the cluster being worked on, unless the whole table is asked for.
- Statuses: `open`, `verify` (looks fixed already: confirm, then close the issue), `needs run`
  (waiting for an experiment), `umbrella` (a collection: split or close when its items are done),
  `later` (set aside by decision; see the last table).

Last rebuilt: 2026-11-19 from the 61 issues open that day; last updated 2026-11-20 after commit `93e4619`.
It replaces the ledger of the 18-lens cleanup (rows 1 to 187): that ledger's open rows became issues
#65 to #81, and its closed rows are in the commit messages of branch `cleanup`.

**Order of work.** Nothing is open: 63 issues closed on 2026-11-19 and 2026-11-20 (the 61 open at the start
and two filed on the way).

## Open issues

None. Every issue of `alejandrofontan/AllFeature-VSLAM` was closed on 2026-11-19 and 2026-11-20. The next
finding gets a GitHub issue first, then a row here:

| # | Lens | Issue | Fix | Label | Status |
|---|---|---|---|---|---|

Closed with work still intended, to re-file when taken up (their closing comments hold the notes):

- **File by file:** #73 and #38 (module headers and member order; the rule is in `review-lenses.md`).
- **RGB-D** (label `rgbd`, closed): #71 (pose optimization rejects about half its matches), #63, #5, #2, #3.
- **Parameter fine-tuning stage:** #27, #65 (`filter_min_iterations`), #72 (per-keypoint sigma).
- **Performance stage:** #13, #78 (`compute_distinctive_descriptors`), #60 (per-model run times in the report).
- **Mac / Core ML:** #58, #61, #59.
- **Developments:** #25 (PoseLib initializer), #26, #24, #10 (multi-map), #23, #41.

## Not an issue (local state)

- VSLAM-LAB must change with the next `allfeature` package release (its repository, not this one): `pixi.toml`
  `copy-package-settings` copies the removed `settings/` folder, `download-superpoint-models` and
  `download-lightglue-models` fetch folders nothing uses, and `baseline_allfeature.py` checks for
  `lightglue_models/aliked-n16.pt` and `superpoint_models/superpoint_v1_fixed.onnx`.
- Local folders `lightglue_models/` and `superpoint_models/` (49 MB each, git-ignored) are used by nothing.

- The working `CMakeLists.txt` and `build/` have `ALLFEATURE_REPORT` ON; the committed default is OFF.
  Set it back before timing experiments.
- Run on ETH `table_3` mono, 2026-11-19 (orb32 + aliked128, `reference_keyframe`): the calibrated geometric
  filter (`4a9e54f`) fitted a model in all 1,727 calls, the feature budgets (`f7204b0`) gave ALIKED 1,024, the
  MapPoint edge cases (`44e7a20`) left 0 keyframes retained after erase; keyframe ATE 5.4 mm, no pose-optimization
  warnings. - Run on TUM `freiburg2_desk` mono, 2026-11-19, same settings, viewer on: clean exit; loop closed (keyframe 374 -> 11,
  447 points matched, 309 fused); the filter fitted a model in all 4,577 calls; 0 keyframes retained after erase
  (185 in map, 406 created); keyframe ATE 8.0 mm (13.8 before the final BA); 32 pose-optimization warnings
  (165 to 197 in the runs of 2026-11-18). Still owed: an `optical_flow` run (`8b646bb`).
- Not exercised by any run: the PnP solver fix of #68 (`cda66fd`). On the next run that loses tracking, check the
  `reloc.*` report entries: a handful of distinct hypotheses per lost frame, not about 20 identical ones.
