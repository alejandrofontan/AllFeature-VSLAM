/**
 * Module: AllFeature-VSLAM - KeyframeInformation.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-24
 * - License: GPLv3 License
 *
 * The keyframe INFORMATION kernel of the system (docs/plans/covisibility_kernel.md): one
 * placecell::PlaceCell store over the keyframes whose similarity kernel answers two
 * questions —
 *   - Tracking::need_new_keyframe: how much of the current view is NOT explained by the
 *     local keyframes (information(), placecell::PlaceCell::unexplained_information);
 *   - LocalMapping::cull_keyframes_information: which alive keyframe every view ever
 *     inserted can spare (place_cell(), placecell::PlaceCell::cull_keyframes).
 * Both read the same kernel, so a keyframe inserted as novel is not a cull candidate on
 * its own account (Schur identity, see placecell's CLAUDE.md).
 *
 * Two backends, selected by the settings key `LocalMapping.InformationKernel`:
 *   - KeyframeInformationMegaLoc    ("megaloc", compiled default): the retrieval store of
 *     the `vpr: megaloc` backend (System::place_cell, MegaLoc image embeddings, cosine
 *     kernel with a common-mode floor, hence centred by default). One store serves
 *     retrieval and information; frames are embedded once per tracked frame and the
 *     embedding is cached in Frame::global_descriptor for the keyframe made from it.
 *     Needs vpr: megaloc — System constructs no component otherwise.
 *   - KeyframeInformationCovisibility ("covisibility"): its own placecell store in ITEM
 *     mode (placecell::PlaceCell::set_items): a keyframe is the set of map points it
 *     observes and the kernel is the cosine of the indicator vectors,
 *     K_ij = |P_i ∩ P_j| / sqrt(|P_i| |P_j|) — PSD, unit diagonal, exactly 0 for keyframes
 *     that share nothing, so raw (uncentred) by default. Independent of the VPR backend
 *     (works with vpr: none). The sets move with the map (points culled, fused, created),
 *     so LocalMapping re-sends every alive keyframe before each cull (refresh()); a culled
 *     keyframe's set is frozen by placecell.
 *
 * Threading: information() runs on the tracking thread; on_keyframe_processed(),
 * refresh() and the cull on the local-mapping thread; clear() from Tracking's reset.
 * placecell serialises its store internally; the map-point snapshots are taken by value
 * under the keyframe / frame ownership rules as everywhere else. The Recorder feed
 * (record_thresholds / record_decision) is a no-op while PlaceCell.Record is off.
 */

#ifndef AF_VSLAM_KEYFRAME_INFORMATION_H
#define AF_VSLAM_KEYFRAME_INFORMATION_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <placecell/placecell.h>

#include "Types.h"

namespace placecell { class MegaLocPlaceCell; }

namespace AF_VSLAM
{

class Frame;
class KeyFrame;
typedef std::shared_ptr<KeyFrame> Keyframe;

// Information a frame's view would add to a set of keyframes
// (placecell::PlaceCell::unexplained_information, on the store's kernel).
struct KeyframeInformationValue
{
    float unexplained{1.0f};      // v in [0,1]: 1 = nothing in the set resembles the view, 0 = fully explained
    int explainers{0};            // keyframes of the set with a usable row in the store
    FrameId best_explainer{0};    // frame_id of the most similar explainer
    float best_similarity{0.0f};  // its similarity on the kernel used (centred or raw)
};

class KeyframeInformation
{
public:
    virtual ~KeyframeInformation() = default;

    // Backend name as written in the settings ("megaloc", "covisibility").
    virtual std::string name() const = 0;

    // Unexplained information of the frame's view given the keyframes in `window`
    // (Tracking's local map), on the kernel keyframe culling marginalises (`centred` as
    // LocalMapping.KeyframeCullingCentred). Read-only for the store. std::nullopt when
    // the view cannot be measured (no image to embed / no tracked map points) or the
    // store cannot compare it (NaN).
    virtual std::optional<KeyframeInformationValue> information(Frame& frame, const std::vector<Keyframe>& window,
                                                                bool centred) = 0;

    // A keyframe finished LocalMapping::process_new_keyframe (observations registered,
    // connections updated, in the map): make the store explain the next frames with it.
    virtual void on_keyframe_processed(const Keyframe& keyframe) = 0;

    // Bring the store up to date with the map before a cull: `alive` are the map's
    // non-bad keyframes. Costs O(total observations) for the covisibility kernel, nothing
    // for MegaLoc (descriptors do not move).
    virtual void refresh(const std::vector<Keyframe>& alive) = 0;

    // System reset (Tracking::reset): forget every keyframe. The MegaLoc backend's store
    // is cleared by PlaceRecognitionMegaLoc::clear (same object), so this is a no-op there.
    virtual void clear() = 0;

    // The store behind the kernel: the culler (LocalMapping::cull_keyframes_information),
    // the Viewer panels and System::SavePlaceCellDiagnostics work on it directly.
    virtual placecell::PlaceCell& place_cell() = 0;
    virtual const placecell::PlaceCell& place_cell() const = 0;

    // Decision history for the plots (placecell Recorder, PlaceCell.Record): the
    // thresholds Tracking's keyframe policy applies (tau = LocalMapping.KeyframeCullingMaxUnexplained,
    // min_information = Tracking.KeyframeMinInformation; every change is kept so plots can
    // draw steps) and its per-frame decision (insertion markers on the information plot).
    // Called once per tracked frame from Tracking::need_new_keyframe.
    void record_thresholds(float tau, float min_information);
    void record_decision(FrameId frame_id, bool inserted, std::optional<float> unexplained, const std::string& reason);
};

// `megaloc`: the retrieval store IS the information store (owned by System, injected here
// and into PlaceRecognitionMegaLoc).
class KeyframeInformationMegaLoc final : public KeyframeInformation
{
public:
    explicit KeyframeInformationMegaLoc(std::shared_ptr<placecell::MegaLocPlaceCell> place_cell);
    ~KeyframeInformationMegaLoc() override;

    std::string name() const override { return "megaloc"; }
    std::optional<KeyframeInformationValue> information(Frame& frame, const std::vector<Keyframe>& window,
                                                        bool centred) override;
    void on_keyframe_processed(const Keyframe&) override {}   // PlaceRecognitionMegaLoc::compute(KeyFrame&) stored it
    void refresh(const std::vector<Keyframe>&) override {}
    void clear() override {}                                  // PlaceRecognitionMegaLoc::clear clears the shared store
    placecell::PlaceCell& place_cell() override;
    const placecell::PlaceCell& place_cell() const override;

private:
    std::shared_ptr<placecell::MegaLocPlaceCell> place_cell_;
};

// `covisibility`: own item-mode store; item ids are MapPoint::ptId.
class KeyframeInformationCovisibility final : public KeyframeInformation
{
public:
    explicit KeyframeInformationCovisibility(const placecell::PlaceCell::Options& options);

    std::string name() const override { return "covisibility"; }
    std::optional<KeyframeInformationValue> information(Frame& frame, const std::vector<Keyframe>& window,
                                                        bool centred) override;
    void on_keyframe_processed(const Keyframe& keyframe) override;
    void refresh(const std::vector<Keyframe>& alive) override;
    void clear() override;
    placecell::PlaceCell& place_cell() override { return *place_cell_; }
    const placecell::PlaceCell& place_cell() const override { return *place_cell_; }

private:
    // Ids of the keyframe's non-null, non-bad map points over every feature type
    // (by-value snapshots of get_map_point_matches).
    static std::vector<placecell::PlaceCell::ItemId> observed_point_ids(const KeyFrame& keyframe);

    std::shared_ptr<placecell::PlaceCell> place_cell_;
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_KEYFRAME_INFORMATION_H
