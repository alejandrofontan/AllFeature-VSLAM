# Documentation, review, gym and report — plan

- **Author:** Alejandro Fontan (planned with Claude, Fable 5.1)
- **Created:** 2026-09-19
- **Status:** in progress, branch `docs`
- **Pinned to:** commit `4577402` (main == dev at the time of writing)

## Goal

Standardise three activities that were happening ad hoc, and add a fourth:

1. **Reference** — what each source file does, for a reader of the code.
2. **Review log** — what the author found while reading each file, dated and pinned to a commit.
3. **Gym log** — what changed because of a failure observed on a VSLAM-LAB sequence, with the
   measured effect.
4. **System report** — the citable LaTeX snapshot of all of the above (separate repo).

The four form a pipeline. Observations go in the review log or the gym log. Facts about how the
code works go in the reference. Derivations go in the report repo. Anything actionable becomes a
GitHub issue (the existing convention).

## Decisions taken (2026-09-19)

| Decision | Choice |
|---|---|
| Report kind | **System report** (AllFeature-VSLAM v2), not a focused keyframe-lifecycle paper |
| Report repo | separate sibling `Baselines/AllFeature-VSLAM-paper/`; GitHub primary remote, Overleaf secondary |
| Where derivations live | in the paper repo, one self-contained `.tex` per derivation; the code repo cites them by file |
| Reference layout | per source file under `docs/reference/`; `allfeature.md` becomes the index + system diagram |
| Line links | keep `src/File.cc#L<n>` links; a resolver script rewrites the line numbers before a docs commit |
| Review log layout | per source file under `docs/review/`, header pinned to a commit, append-only |
| Review log visibility | public chapter on the docs site |
| Doxygen appendix | not now |
| First reference pass | the five pipeline files: System, Tracking, LocalMapping, LoopClosing, Viewer |
| Gym set | `eth/table_3` only, to start |
| Benchmark set | issue #18's 100 sequences, run at milestones only |
| CLAUDE.md | migrated in the same pass: investigations move to `docs/`, CLAUDE.md keeps guidance |
| Branching | `main` fast-forwarded to `dev`, work on `docs` off `main`, one commit per step |
| Navigation | MkDocs Material on GitHub Pages, built from `docs/` by a GitHub Action on push to `main` |

## Layout

```
docs/
  index.md                  landing page of the site (system overview + clickable system diagram)
  reference/<File>.md       what each source file does (format: docs/reference/README.md)
  review/<File>.md          dated reading notes per file, pinned to a commit (format: docs/review/README.md)
  gym/protocol.md           how a gym cycle is run and measured
  gym/board.md              one row per gym sequence: current best numbers + the entries that touched it
  gym/entries/<date>_<slug>.md   one entry per measured change
  notes/<date>_<slug>.md    dated investigations (migrated from CLAUDE.md and profiling.md)
  plans/<slug>.md           design plans (this file, segmentation, RGB-D depth integration)
  tools/                    resolve_links.py, review_checklist.py, gym_compare.py
allfeature.md               index + system-level diagram
CLAUDE.md                   guidance only; points at docs/
mkdocs.yml                  site definition
.github/workflows/docs.yml  build + deploy the site on push to main
```

Sibling repo `Baselines/AllFeature-VSLAM-paper/`: `main.tex` with the site's chapter structure,
`sections/`, `appendix/` (derivations), `figures/` with the scripts that generate them from
`VSLAM-LAB-Evaluation/`.

## The four artifacts

### 1. Reference (`docs/reference/`)

Per file: a call-graph list, a mermaid flow diagram, then one entry per function with a heading,
the signature, bullets (what it does, called from, why it differs from stock ORB-SLAM2 when it
does), the settings keys it reads, and `src/File.cc#L<n>` links. Format in
`docs/reference/README.md`; the `/document-file` command (`.claude/commands/document-file.md`)
produces a page in that format. `docs/tools/resolve_links.py` rewrites line numbers from the
link's search pattern before a docs commit.

Rule: a code change that alters behaviour updates the file's reference page in the same commit,
or the commit message says why not.

### 2. Review log (`docs/review/`)

Per file: header with the commit hash and date of the review, a checklist of every function with a
state (`unread / ok / question / bug / redesign`), then dated notes tagged with where each finding
went: `→ #<issue>`, `→ reference`, `→ paper:<file>.tex`, `→ gym`. Append-only: a note is never
rewritten when the code changes; a new dated note supersedes it. `docs/tools/review_checklist.py`
regenerates the function checklist from the source file, preserving existing states.

Rule: a review session on a file produces one commit touching `docs/review/<File>.md` plus zero or
more issues. Nothing else.

### 3. Gym log (`docs/gym/`)

One entry per change, not per run. Entry: trigger (sequence, config, commit, failure signature),
hypothesis, change (commit or PR, files touched), before/after table with fixed columns, verdict
(kept / reverted / kept with a known cost), issue link. Board: one row per gym sequence with its
current best numbers and the entries that touched it.

Ground rules (details in `docs/gym/protocol.md`): three or more runs per configuration; one change
per entry against the same baseline commit; fixed gym configs `configs/gym_*.yaml` in VSLAM-LAB;
`--overwrite` always. A change kept in the gym gets a benchmark regression check before it counts
as final, and the entry records that check. `docs/tools/gym_compare.py` prints the before/after
table from two experiment folders; the hypothesis and change text stay manual.

### 4. System report (paper repo)

Same chapter structure as the docs site. Frozen, citable snapshot of the living site. Figures are
the site's diagrams exported as PDF (mermaid CLI) plus plots generated by committed scripts.

## Navigation (site)

Sidebar chapters: Overview, Pipeline (one page per thread), Components, Settings, Gym, Review,
Notes, Plans. Three zoom levels, all clickable: system diagram → component page, component diagram
→ function entry, function entry → source line. Settings keys link both ways between the reference
page that reads them and the settings table. Short `.mp4` clips under 10 MB per component page,
committed. One narrated walkthrough on YouTube, recorded last.

## Migration of existing files

| Source | Destination |
|---|---|
| `CLAUDE.md` § RGB-D Depth Integration Opportunities | `docs/plans/rgbd_depth_integration.md` |
| `CLAUDE.md` § Optimizer.cc / Optimizer.h Review (2026-08-09) | `docs/review/Optimizer.md` (first entry) |
| `CLAUDE.md` § Build-Time Memory/Swap Investigation (2026-08-09) | `docs/notes/2026-08-09_build_memory_swap.md` |
| `CLAUDE.md` § Compiler Warnings Audit (2026-08-09) | `docs/notes/2026-08-09_compiler_warnings_audit.md` |
| `CLAUDE.md` § Tracking-Lost Audit (2026-08-10) | `docs/review/Tracking.md` (first entry) |
| `CLAUDE.md` § Stop-Induced Keyframe Runaway Investigation (2026-08-12) | `docs/notes/2026-08-12_keyframe_runaway_nsavp.md` |
| `CLAUDE.md` § Scattered Tracking-Loss Investigation (2026-08-13) | `docs/notes/2026-08-13_scattered_tracking_loss_nsavp.md` |
| `CLAUDE.md` § Visual Place Recognition (2026-08-28) | `docs/notes/2026-08-28_place_recognition_megaloc.md` |
| `CLAUDE.md` § Information-Based Keyframe Insertion (2026-09-03) | `docs/notes/2026-09-03_keyframe_information_policy.md` |
| `CLAUDE.md` § placecell diagnostics (2026-09-05) | `docs/notes/2026-09-05_placecell_diagnostics.md` |
| `profiling.md` protocol | `docs/gym/protocol.md` (generalised) |
| `profiling.md` results log | `docs/gym/entries/2026-08-16_*.md` (one entry per item) |
| `segmentation.md` | `docs/plans/segmentation.md` |
| `TODO.md` | open rows → issues; file deleted |
| `allfeature.md` § Tracking, § LoopClosing | `docs/reference/Tracking.md`, `docs/reference/LoopClosing.md` |

Migrated text is moved verbatim under a short provenance header (source, original date). Rewrites
happen afterwards as ordinary reference/review work, not as part of the move.

## Order of work

1. This plan, the `docs/` skeleton, the file moves, `TODO.md` → issues. *(commit)*
2. CLAUDE.md migration and slimming. *(commit)*
3. Reference format (`docs/reference/README.md`), `/document-file` command, `resolve_links.py`,
   split of `allfeature.md`. *(commit)*
4. Reference pages for System, Tracking, LocalMapping, LoopClosing, Viewer. *(commit per page)*
5. Review log format, `review_checklist.py`, seeded pages for the five pipeline files + Optimizer. *(commit)*
6. Gym: protocol, board, migrated entries, `gym_compare.py`, `configs/gym_*.yaml` in VSLAM-LAB. *(commit)*
7. MkDocs site: `mkdocs.yml`, pixi `docs` feature, GitHub Action, landing page, settings page. *(commit)*
8. Paper repo skeleton. *(separate repo)*

## Open

- Video clips and the narrated walkthrough: after the first reference pass, not in this pass.
- Second gym sequence (a driving sequence with stops, NSAVP `R0_FA0` was the candidate).
- Whether the parameter-tuning work of issues #3 and #27 is logged as gym entries (yes by design;
  an autotuner writing entries automatically is a separate decision).
