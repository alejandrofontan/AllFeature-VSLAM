# Review lenses

## A. Correctness — does it work?

1. **Memory safety & segmentation faults** — Does it crash: invalid pointers, lifetimes, bounds, destruction order.
2. **Thread safety** — Does concurrency corrupt it: lock discipline, shared state, teardown order, determinism.
3. **Bugs (logic correctness)** — Does it do the wrong thing: control flow, indices, state updates, untrue assumptions.
4. **Numerical & algorithmic correctness** — Does the maths hold: units, thresholds, precision, degeneracies, gauge.
5. **Error handling & failure modes** — What happens when it fails: how failure is signalled, bounded and survived.
6. **Data & IO contracts** — Does what it writes and reads still match what the other side expects.

## B. Cost — does it work well?

7. **Performance** — Does it spend more time or memory than the work requires.
8. **Observability** — Could you diagnose a failed run from its output alone.

## C. Maintainability — can you change it?

9. **Cleanup (deletion)** — What can be removed outright: dead code, dead flags, dead files, untrue comments.
10. **Syntax & style** — Does it follow the project's stated conventions and modern C++ idiom.
11. **Encapsulation** — Does each class own its state and hide what nobody else needs.
12. **Extensibility** — What it costs to add the next feature type, backend, sensor or input source.
13. **Configuration hygiene** — Is every tunable defaulted, read, documented and tabulated the one agreed way.
14. **Documentation sync** — Do the reference pages still describe the code as it is now.

## D. Reach — does it work elsewhere?

15. **Portability** — Does it behave the same on both platforms and every execution provider.
16. **Build system & packaging** — Do all the option combinations configure, build and install cleanly.

## E. Direction & discipline

17. **Improvements (behaviour & capability)** — What the system should do and doesn't yet.
18. **Verification** — How a finding becomes a change you can prove is safe.

## Findings table

| # | Lens | Issue | Fix | Label | Status |
|---|---|---|---|---|---|
| row number, stable across re-issues of the table so "do 7" keeps meaning the same thing | which of the 18 lenses caught it | what is wrong, stated as a fact about the code, with the `file:line` | the concrete change, specific enough to act on without rediscovering it | one of the 14 labels | `open` / ✅ done / ❌ won't fix / deferred, with the reason when it isn't open |

**Labels:** `bug`, `latent bug`, `improvement`, `style`, `naming`, `inconsistency`, `cleanup`,
`altitude`, `verification`, `dead code`, `duplication`, `magic number`, `fragility`,
`needs analysis`.

**Conventions.** Row numbers are never reused — a fixed row stays in the table marked ✅ rather
than being dropped, so `do 12` never points at two different things across a session. Any fix that
is not byte-identical says so in the Fix cell; those cells are what the verification list is built
from at the end of the file.

### Blank

| # | Lens | Issue | Fix | Label | Status |
|---|---|---|---|---|---|
| 1 |  |  |  |  | open |
| 2 |  |  |  |  | open |
| 3 |  |  |  |  | open |

## Working constraints

1. **Be concise, explain simply.** Short answers. Plain language over jargon, and a worked
   example over a definition. No preamble, no restating the question, no listing options that
   won't be pursued — give the recommendation. If a point needs more than a few lines, it is
   probably two points.

2. **After every final edit, re-issue the table and say what's next.** Show it filtered to the
   open rows only, then one line recommending the next row to take and why. "Final edit" means
   the last change of a fix, not each intermediate one — the table comes after the fix is
   complete and builds, not after every file touched.

3. **Take the pattern from the already-cleaned files.** Before reviewing a new section, read the
   ones already done and carry their decisions forward rather than re-deriving them:

   `include/Frame.h`, `src/Frame.cc`, `include/KeyFrame.h`, `src/KeyFrame.cc`

   They are the reference for naming, for structure, for idiom and for section banners and member ordering.

   A decision already made there is settled — don't re-open it, extend it. If the new file
   genuinely needs to differ, say why in one line and record it; if it reveals the earlier
   decision was wrong, raise that as its own row rather than quietly diverging.

## File conventions

Settled on 2026-11-20 (issues #74, #73, #38). Applied to a file when that file gets its cleanup pass,
not in one sweep.

**Include order** in a `.cc` / `.cpp`: four blocks, one blank line between them, each block sorted
alphabetically.

1. The file's own header (`#include "KeyFrame.h"` in `KeyFrame.cc`): first, so it is proven to
   compile on its own.
2. The standard library (`<algorithm>`, `<mutex>`, ...).
3. Third-party libraries (OpenCV, Eigen, PoseLib, yaml-cpp, placecell, Pangolin, g2o, ONNX Runtime).
4. Project headers (`"Converter.h"`, `"onnx/FeatureModels.h"`, ...).

A header (`.h`) follows the same order without block 1.

**Module header** at the top of every source and header file, as in `src/onnx/FeatureModels.cpp`:

```cpp
/**
 * Module: AllFeature-VSLAM - <path under src/ or include/>
 * - Author: <name>
 * - Assisted by: <if any>
 * - Version: <major.minor>
 * - Created: <date of the file's first commit>
 * - Updated: <date> (<what changed>)
 * - License: GPLv3 License
 */
```

It replaces an IDE stub (`// Created by ...`); a file that carries the ORB-SLAM2 GPL banner keeps
the banner below it. "Created" comes from `git log --diff-filter=A --format=%as -- <file>`.

