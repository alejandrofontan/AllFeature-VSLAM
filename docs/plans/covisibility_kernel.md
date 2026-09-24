# Covisibility kernel — host-supplied measurements for the information culler

- **Author:** Alejandro Fontan (planned with Claude, Fable 5.1)
- **Created:** 2026-09-24
- **Status:** implemented, unrun — steps 2–5 done 2026-09-24 (placecell branch `item-mode` + AllFeature working tree, both uncommitted, library builds clean); step 1 skipped by decision; step 6 (gym) waits for the first runs
- **Pinned to:** AllFeature `0c77449` (`dev`), placecell `6ec557f` (`main`)

## Goal

Let placecell build its information kernel from measurements the SLAM system produces, instead
of the MegaLoc embedding, starting with the number of map points two keyframes share. The
kernel must keep tracking the map as it changes (fusion, point culling, loop correction), not only
at keyframe creation. MegaLoc stays the retrieval descriptor for relocalization and loop
candidates; only the *information* side (keyframe insertion and culling) becomes geometric.

## Why this fits placecell as it is

Represent keyframe `i` by the indicator vector over the map's points (1 where `i` observes the
point). The shared-point count `c_ij` is the dot product of two such vectors and

    K_ij = c_ij / sqrt(n_i n_j)        (n_i = number of good map points of keyframe i)

is their cosine. `K` is a Gram matrix, therefore PSD, with unit diagonal and entries in
`[0, 1]`: exactly the contract `cull_keyframes` and `unexplained_information` marginalise today.
Everything the culler derives (`v_i = 1/(K_AA^-1)_ii`, the rank-one downdates, the Schur identity
between the insertion query and the post-insertion unique information) carries over unchanged.
Two things differ from the MegaLoc kernel:

- **no common-mode floor**: unrelated keyframes score exactly 0, so centring is unnecessary
  (placecell issue #4 made concrete). With centring off, the first-cull degeneracy of issue #5
  (centring set == alive set) does not arise for this kernel;
- **the "descriptor" is mutable**: a keyframe's point set changes after insertion, so entries
  between existing views must be updatable. The culler snapshots the kernel at every call and is
  unaffected; what breaks is `add()`'s one-shot row, the append-only contract in the header, and
  the incrementally maintained row sums.

The measured proxy quality is already on record: the placecell notes report an
information-weighted covisibility cosine correlating 0.985 with the exact BA mutual-information
kernel (`tools/colmap_information_kernel.py`). Step 1 below re-measures this for the unweighted
cosine.

## Decisions (2026-09-24)

| Question | Decision |
|---|---|
| Store split on the host | two stores: `MegaLocPlaceCell` keeps retrieval; a second `PlaceCell` in the new mode holds the information kernel |
| Measurement | raw shared map-point count, pooled over feature types (weights and per-type kernels later) |
| Normalization | cosine `c / sqrt(n_i n_j)`; centring off by default for this kernel |
| Data flow | **item sets**: the host sends each view's observed point ids; placecell owns the kernel, the normalisation and an inverted index |
| History rows | placecell freezes the culled view's item set; its row is recomputed exactly against the alive views' current sets |
| Insertion query | the current frame's matched map points (Tracking) form the query item set; insertion becomes geometric, `low_overlap` / `weak_tracking` retire later, after a gym entry |
| Combination with MegaLoc | none; geometric only (product / blend noted as future options) |
| Tau | one key and slider as today; per-kernel compiled default and dev-YAML value, tuned in the gym |
| Refresh | `cull_keyframes_information` re-sends every alive keyframe's good point ids before each cull; the new keyframe is also sent in `process_new_keyframe` so it explains at once |
| Validation | offline first: covisibility cosine from a COLMAP model vs the mutual-information kernel, before any online code |
| Switchable | `LocalMapping.InformationKernel: megaloc \| covisibility`; MegaLoc path unchanged for A/B |
| Empty views | a view with no items has an unusable (NaN) row, WARN once; same treatment as a descriptor-size mismatch |
| Bindings | `set_items` and the item query exposed in nanobind; example executables untouched |
| Refactor scope | item sets land as a **third mode** next to descriptors and `set_kernel` (two-modes-never-mixed pattern); the strategy unification of issue #1 is a follow-up |
| Item ids | the map point's current `ptId`, whole set re-sent at each refresh; no replacement-chain resolution |
| Host owner | a new `KeyframeInformation` component owned by `System`; `PlaceRecognition` returns to retrieval only |

## Step 1 — offline validation (no SLAM run)

`tools/colmap_information_kernel.py` already reads which unique point ids every registered image
observes (`ImageInfo.pids`). Add a second output next to `kernel.npy`:

- `covisibility_kernel.npy`: `K_ij = |P_i ∩ P_j| / sqrt(|P_i| |P_j|)` over the same registered
  images, same `ids.csv` row order, float32, unit diagonal;
- one stdout line with its eigenvalue range (expected PSD) and the unique-information
  distribution `1/(K^-1)_ii` on the raw kernel, which gives the tau scale of this kernel.

Generalise `tools/compare_kernels.py` to accept two kernel directories (today it takes `D.npy`
plus one directory) and run it on ETH `table_3`: covisibility vs mutual information, and
covisibility vs MegaLoc. Then `kernel_demo <out>/covisibility_kernel.npy --kind similarity --ids
<out>/ids.csv --raw` at a few taus to see which frames survive. Record the numbers in a
`docs/notes/` page; they set the compiled default of tau for the `covisibility` kernel.

## Step 2 — placecell: item-set mode

New public API on `PlaceCell` (`include/placecell/placecell.h`), following the kernel-only mode's
shape: the first call decides the mode, `clear()` resets it, and the other modes' entry points are
refused with an ERROR log.

```cpp
using ItemId = std::uint64_t;

struct ItemOptions
{
    // cosine is the only normalisation of the first version; the field exists so Jaccard can
    // be added without an API change
    enum class Normalization { cosine } normalization{Normalization::cosine};
};

// Store or replace the item set of a view (a keyframe's map-point ids). A new id becomes a
// new row (all alive views become its explainers); a known id has its set replaced and the
// affected kernel entries updated. Refused on a culled view (its set is frozen), on a
// descriptor-backed or kernel-only store, and logged once for an empty set (unusable row).
// Idempotent for an identical set. Returns the internal id.
InternalId set_items(ExternalId id, std::vector<ItemId> items);

// Item set of a stored view (frozen for culled views); nullptr for an unknown id or another mode.
const std::vector<ItemId>* items(ExternalId id) const;

// Unexplained information of a view NOT in the store, given by its item set (the current
// frame's tracked map points), over the alive views or the alive views among `window`.
// Same maths as the descriptor overload; centred defaults to false for this kernel.
Information unexplained_information(const std::vector<ItemId>& items,
                                    const std::vector<ExternalId>* window = nullptr,
                                    bool centred = false) const;

bool item_mode() const;
```

Internals (`src/placecell.cpp`), all under the existing single mutex with the linear algebra
outside it:

- per view: a sorted `std::vector<ItemId>`; an inverted index `unordered_map<ItemId,
  vector<InternalId>>`; an integer shared-count matrix `counts_` (n×n) and `norms_[i] = |P_i|`.
- `set_items(id, new)`: diff sorted old/new; for every removed item, decrement `counts_(i, j)` for
  each `j` in the item's posting list and remove `i` from it; for every added item, the reverse.
  Cost O(|Δ| × average views per item). Culled views never change, so their frozen sets stay
  in the index and keep contributing to `counts_(h, a)` as alive sets move.
- `kernel_` is produced from `counts_` and `norms_` when a snapshot is taken (`snapshot`,
  `cull_keyframes`, `kernel()`, `dump`): `K_ij = counts_(i,j) / sqrt(norms_i norms_j)`, NaN row
  when a norm is 0. This is O(n²) per snapshot, the same order as the existing centring pass.
  The incremental row sums are not maintained in this mode; if `centred = true` is requested,
  the sums are computed from the snapshot instead.
- the item query: for each query item, walk its posting list and count hits per view, then
  `k_xA(a) = hits_a / sqrt(|Q| n_a)`, `k_xx = 1`, and the same jittered LDLT solve as today.
  Cost O(Σ posting-list lengths) plus the |explainers|³ solve; no sweep of the kernel.
- `cull_keyframes` needs no change: it reads the snapshot. The view culled through the callback
  is marked culled before the callback runs, so a later `set_items` on it is refused and its set
  is frozen by construction. `set_culled` (host-side removals) freezes the same way.
- events: `on_set_items(id, internal, added, removed, empty)` next to `on_add` in
  `placecell_events.cpp` (DEBUG per call, WARN once for an empty set); profiler rows `set_items`
  and `unexplained_information_items`, declared in the constructor.
- header contract: document the third mode, the frozen-set rule, and that `descriptor()` is
  nullptr and `add()` / `set_kernel()` are refused in item mode.
- `dump`/`viz`: unchanged (they read snapshots); `views.csv` gains an `items` column (set size).
- bindings: `set_items(id, ndarray[uint64])`, `items(id)`, `unexplained_information(items,
  window, centred)`; re-exported in `__init__.py`.
- smoke test: extend `synthetic_demo` with an item-set twin only if Step 1's COLMAP data is not
  enough; otherwise a Python check on the COLMAP incidence (build the store with `set_items` from
  `ImageInfo.pids`, compare `kernel()` with `covisibility_kernel.npy`, cull, compare survivors
  with `kernel_demo`).

Not in scope: `add_row` / host-supplied scalar rows (issue #2 as stated), weights per item,
Jaccard, the strategy refactor (issue #1). Each gets its own issue when this lands.

## Step 3 — AllFeature: `KeyframeInformation` component

New pair `include/KeyframeInformation.h`, `src/KeyframeInformation.cc` (module header docstring,
snake_case, trailing underscores), owned by `System` and handed to `Tracking`, `LocalMapping` and
the `Viewer`:

```cpp
class KeyframeInformation
{
public:
    // Unexplained information of the frame's view given `window` (Tracking's local map)
    virtual std::optional<KeyframeInformationValue> information(Frame& frame,
                                                                const std::vector<Keyframe>& window) = 0;
    // Bring the store up to date with the map (item mode: re-send every alive keyframe)
    virtual void refresh(const std::vector<Keyframe>& alive) = 0;
    // Register a keyframe (item mode: its current good points; megaloc: no-op, the retrieval
    // store already holds the descriptor) and reconcile externally removed keyframes
    virtual void on_keyframe_processed(const Keyframe& keyframe) = 0;
    virtual void set_culled(FrameId frame_id) = 0;
    virtual placecell::PlaceCell::CullReport cull(const placecell::PlaceCell::CullParameters&,
                                                  const placecell::PlaceCell::CullCallback&,
                                                  const std::vector<FrameId>* window) = 0;
    // Recorder feed (moved here from PlaceRecognition)
    virtual void record_thresholds(float tau, float min_information) = 0;
    virtual void record_decision(FrameId, bool inserted, std::optional<float>, const std::string& reason) = 0;
    // The store behind it, for the Viewer panels and System::SavePlaceCellDiagnostics
    virtual const placecell::PlaceCell* place_cell() const = 0;
};
```

Two backends behind `LocalMapping.InformationKernel`:

- `megaloc`: wraps the retrieval `MegaLocPlaceCell` (moved out of `PlaceRecognitionMegaLoc`:
  `keyframe_information`, `record_keyframe_thresholds`, `record_keyframe_decision`); behaviour
  identical to today, so gym A/B runs stay possible.
- `covisibility`: owns a `placecell::PlaceCell` in item mode (`PlaceCell.*` options apply to it
  too; with `megaloc` there is one store, with `covisibility` there are two and both print their
  profile at shutdown). Item ids are `MapPoint::ptId`.

Host wiring:

- `System`: construct the component from the settings key after the VPR backend; pass it to
  `Tracking`, `LocalMapping`, `Viewer`; `GetPlaceCell()` returns the *information* store (what the
  panels and the dump show); `Shutdown`/`SavePlaceCellDiagnostics` unchanged otherwise.
- `Tracking::need_new_keyframe`: the query is the current frame's map points that are non-null,
  non-bad, non-outlier and have at least one observation, collected as `ptId`s after
  `track_local_map` (the same points `update_local_keyframes` votes with). `information(frame,
  local_keyframes_)` replaces `place_recognition_->keyframe_information(...)`; the three bands,
  the `decide` lambda and the diagnostic line stay. `Frame::global_descriptor` remains for the
  `megaloc` backend only.
- `LocalMapping::process_new_keyframe`: after `update_connections`, `on_keyframe_processed
  (current_keyframe_)` so the new keyframe explains the next frames immediately (closes caveat 1
  of `docs/notes/2026-09-03_keyframe_information_policy.md` for the item mode).
- `LocalMapping::cull_keyframes_information`: `refresh(map_->GetAllKeyFrames() minus bad)` with
  each keyframe's good point ids (`get_map_point_matches` per feature type, by-value snapshots),
  then the existing reconciliation and cull through the component. One sweep per keyframe cycle,
  O(total observations); measure it in the gym entry.
- `PlaceRecognition` / `PlaceRecognitionMegaLoc`: remove the information and recorder hooks;
  `PlaceRecognitionNone` shrinks accordingly. `vpr: none` + `covisibility` becomes a valid
  combination (information without retrieval); `vpr: none` + `megaloc` keeps today's behaviour
  (no information, tracking triggers only).
- Settings: `LocalMapping.InformationKernel` (`megaloc`, dev YAML `covisibility`), and the tau
  default per kernel: keep `KeyframeCullingMaxUnexplained` as the one key; when the key is absent
  the compiled default depends on the kernel (value from Step 1). `KeyframeCullingCentred`
  defaults to 0 for `covisibility`.
- Docs: reference pages `Tracking`, `LocalMapping`, `System` in the same commits; a new
  `docs/reference/KeyframeInformation.md`; `docs/tools/settings_keys.py` picks the new key up.

## Behaviour to expect and watch

- **Scale of v.** With a single explainer `v = 1 - s²`: a frame sharing 200 of its 300 points
  with a keyframe holding 600 gives `s ≈ 0.47`, `v ≈ 0.78`. Tau and `KeyframeMinInformation`
  need their own defaults for this kernel (Step 1 gives the culler's; the insertion band needs a
  logged run with `Tracking.LogKeyframeInformation: 1`).
- **Moving history.** The invariant "every view ever inserted stays within tau of the alive ones"
  is re-evaluated against a kernel that changes; `history_over_budget` will move without tau
  moving. The existing over-budget slack already tolerates rows above tau.
- **Tracking-dependent insertion.** A frame with few tracked points has a small item set and a
  high `v`, so poor tracking reads as novelty. This is the intended replacement for
  `weak_tracking` / `low_overlap`; retire those only after a gym entry shows the bands cover them.
- **Sparser kernel.** Far keyframes are exact zeros; `K_AA` is better conditioned than MegaLoc's
  and `scope: local` matters less.
- **Threads.** `refresh` and `on_keyframe_processed` run on the local-mapping thread, the query on
  the tracking thread, `set_culled` from the culler; placecell serialises internally. Snapshots of
  map-point vectors are taken by value under the keyframe mutexes as everywhere else.

## Order of work

1. ~~Step 1 offline validation~~ — **skipped** (decided 2026-09-24: go straight to the item mode; the COLMAP comparison can still be done later with `tools/compare_kernels.py` once a covisibility dump exists).
2. **Done 2026-09-24** — placecell item mode: header contract, `set_items`, inverted index, lazily materialised kernel, item query on the shared `explainers_locked`/`query_system_locked`/`solve_information` path, `on_set_items`, profiler rows, `views.csv` items column; plus `usable_rows` (a NaN culprit row no longer disables the whole culler). Verified by a scratch parity harness against a descriptor store of indicator vectors (kernels, queries, culler scores to ~1e-7; Schur identity; incremental == fresh; frozen culled set; empty view isolated), `synthetic_demo`, `kernel_demo`. *(placecell commit pending)*
3. **Done 2026-09-24** — bindings `set_items` / `items` / `item_mode` / `unexplained_information_items` (+ `PlaceCell.invalid_id`), smoke-tested from Python in placecell's pixi env; the COLMAP-incidence check moved out with step 1. *(same placecell commit)*
4. **Done 2026-09-24** — `include/KeyframeInformation.h` / `src/KeyframeInformation.cc` (`KeyframeInformationValue`, abstract `KeyframeInformation`, `KeyframeInformationMegaLoc`); `PlaceRecognition` lost `keyframe_information` / `record_keyframe_*` and the value struct; `Tracking` and `LocalMapping` take the component through `set_keyframe_information`; `System` builds it after the VPR backend, `GetPlaceCell` returns the information store. The `megaloc` path is the old code moved, so it stays behaviour-neutral. *(AllFeature commit pending, with the submodule bump)*
5. **Done 2026-09-24** — `KeyframeInformationCovisibility` (own item-mode store, item ids = `MapPoint::ptId`); Tracking queries with the frame's non-null / non-bad / non-outlier points (empty → no measure); `process_new_keyframe` registers the keyframe, `cull_keyframes_information` refreshes every alive keyframe before the cull; `LocalMapping.InformationKernel` (`megaloc` compiled default, dev YAML `covisibility`) with per-kernel defaults tau 0.7 / centred 0 when the keys are absent; `SavePlaceCellDiagnostics` dumps a distinct retrieval store into `retrieval_megaloc/`; reference pages `KeyframeInformation` (new), `Tracking`, `LocalMapping`, `System`, `settings`, index. Not run yet: the 0.7 tau and the 0.05 redundancy band on this kernel are starting points for step 6. *(same AllFeature commit)*
6. Gym: `megaloc` vs `covisibility` on `eth/table_3`, then the `low_overlap` / `weak_tracking` retirement as its own entry. *(gym entries)*

## Open

- Weighted items (observation count, depth-verified, keypoint information) and per-feature-type
  kernels: after the raw count has run.
- Jaccard as a selectable normalisation.
- Product / blend with the MegaLoc kernel.
- The strategy unification of the three modes (placecell issue #1) and host-supplied scalar rows
  (issue #2).
- Whether the item mode should also serve VSLAM-LAB's offline selection (`rgb_placecell`) from a
  COLMAP model: the Python path of Step 3 already makes it possible.
