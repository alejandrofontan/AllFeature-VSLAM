# Review log — format

One page per source-file family, same split as `docs/reference/`: `docs/review/Tracking.md` covers
`src/Tracking.cc`, `src/Tracking_aux.cc`, `include/Tracking.h`. A page is the author's dated
reading notes: what was found, when, at which commit, and where the finding went. It is
**append-only**: a note is never edited when the code changes; a new dated note supersedes it.

The reference page says what the code does. The review page says what the author thought of it.
The gym says what happened when it ran. Anything actionable becomes a GitHub issue.

## Page skeleton

````markdown
# Review — Tracking

- **Sources:** `src/Tracking.cc`, `src/Tracking_aux.cc`, `include/Tracking.h`
- **Reviewer:** Alejandro Fontan
- **Last reviewed at:** `4577402` (2026-09-19) — one line on how far the read got

## Checklist

| Function | Source | State | Notes |
|---|---|---|---|
| `Tracking::track` | [Tracking.cc#L152](…) | ok | |
| `Tracking::relocalize` | [Tracking.cc#L1091](…) | question | candidate ordering, see 2026-09-19 |

## Notes

### 2026-09-19 — relocalization candidate ordering (at `4577402`)

- `relocalize` tries candidates in retrieval order and stops at the first accepted hypothesis; a
  later candidate with more inliers is never seen. → #NN
- the `RelocInliersMedium` band is documented as "enter the narrow-window escalation" but the
  code … → reference
````

## Checklist

Generated and refreshed by `docs/tools/review_checklist.py` from the `- **Sources:**` line:

```bash
python docs/tools/review_checklist.py --new LocalMapping src/LocalMapping.cc src/LocalMapping_aux.cc include/LocalMapping.h
python docs/tools/review_checklist.py docs/review/LocalMapping.md   # after the code changed
python docs/tools/review_checklist.py --all
```

The tool lists every function defined in the `.cc`/`.cpp` sources, keeps the `State` and `Notes`
cells of functions still present, adds new ones as `unread`, and keeps vanished ones as `gone` so
their notes survive. Edit only the `State` and `Notes` cells by hand.

| State | Meaning |
|---|---|
| `unread` | not read yet at the pinned commit |
| `ok` | read, nothing to do |
| `question` | read, something unclear or worth a second look; the note says what |
| `bug` | read, wrong; an issue exists (`→ #NN` in Notes) |
| `redesign` | read, works but should be built differently; an issue or a plan exists |
| `gone` | no longer defined in the sources; row kept for its notes |

`Notes` in the table is one short phrase, at most a pointer to a dated note below.

## Notes

- **One H3 per session and topic**, titled `### <date> — <topic> (at \`<commit>\`)`. The commit is
  the one that was *read*, which is not always `HEAD` at the time of writing.
- **Bullets.** A finding is one bullet: what was observed, why it matters if not obvious, and a
  tag saying where it went.
- **Tags**, at the end of the bullet:
  - `→ #NN` — filed as an issue (bugs, redesigns, missing features);
  - `→ reference` — the finding is a fact about the code that the reference page now states;
  - `→ paper:<file>.tex` — feeds a section or derivation of the system report;
  - `→ gym` — needs a run to decide; a gym entry will follow;
  - `→ fixed <commit>` — fixed on the spot in the same session (rare; prefer an issue).
- **Never rewrite.** If a later read contradicts an earlier note, write a new dated note that says
  so and links the old one by its heading. History is the point.
- **Maths goes to the paper repo.** A derivation longer than a few lines is a `.tex` file in
  `AllFeature-VSLAM-paper/appendix/`; the note keeps the statement and links it.

## Session recipe

1. `git rev-parse --short HEAD`, refresh the checklist.
2. Read the functions in call-graph order (the reference page's `## Call graph` is the order).
3. Update the `State` cells as you go; write the dated note at the end of the session.
4. File issues for `bug`/`redesign` rows; put the issue number in the note and the table.
5. Update `- **Last reviewed at:**`.
6. One commit: `Review Tracking: <topic> (at <commit>)`, touching this page only (plus issues).
