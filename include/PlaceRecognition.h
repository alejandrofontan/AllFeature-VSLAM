/**
 * Module: AllFeature-VSLAM - PlaceRecognition.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 *
 * Visual place recognition (VPR) backend interface (issue #17, stage 2). One object
 * owns everything retrieval-related: computing a frame's/keyframe's global
 * descriptor, a similarity score, and the two candidate queries (relocalization,
 * loop detection). The downstream geometric verification (feature matching, PnP,
 * Sim3) is backend-agnostic and uses the local feature named by
 * verification_feature() (settings key `feature_vpr`). A new backend (another
 * global descriptor / retrieval method) plugs in by implementing this interface
 * and registering in System's constructor.
 *
 * Backends (selected by the `vpr:` settings key, resolved in System's constructor):
 *   - PlaceRecognitionMegaLoc  MegaLoc global image descriptor via placecell (default)
 *   - PlaceRecognitionNone     disabled: no loop closing, no relocalization
 */

#ifndef AF_VSLAM_PLACE_RECOGNITION_H
#define AF_VSLAM_PLACE_RECOGNITION_H

#include <memory>
#include <string>
#include <vector>

#include "Types.h"

namespace AF_VSLAM
{

class Frame;
class KeyFrame;
typedef std::shared_ptr<KeyFrame> Keyframe;

class PlaceRecognition
{
public:
    explicit PlaceRecognition(const FeatureType verification_feature)
        : verification_feature_(verification_feature) {}
    virtual ~PlaceRecognition() = default;

    // Backend name as written in the settings ("megaloc", "none").
    virtual std::string name() const = 0;

    // The single gate every consumer checks. When false the system runs without VPR:
    // no loop closing, no relocalization (a tracking loss is then permanent).
    virtual bool is_active() const = 0;

    // Local feature used to geometrically verify retrieved candidates (reloc PnP,
    // loop Sim3).
    FeatureType verification_feature() const { return verification_feature_; }

    // True when the backend embeds IMAGES (not local descriptors): Frame/KeyFrame
    // must then keep their image until compute() has run on them.
    virtual bool needs_image() const { return false; }

    // Global descriptor of a query frame / a keyframe (no-op when inactive or
    // already computed). compute(KeyFrame&) may run off the tracking thread
    // (LocalMapping::ProcessNewKeyFrame) and releases the keyframe's image afterwards.
    virtual void compute(Frame& frame) = 0;
    virtual void compute(KeyFrame& keyframe) = 0;

    // Keyframe database
    virtual void add(const Keyframe& keyframe) = 0;
    virtual void erase(const Keyframe& keyframe) = 0;
    virtual void clear() = 0;

    // Similarity between two keyframes' global descriptors (backend-specific scale:
    // cosine in [-1,1] for MegaLoc). 0 if either is missing.
    virtual float score(const KeyFrame& a, const KeyFrame& b) const = 0;

    // Loop candidates for a keyframe: not covisible with it, similar enough,
    // accumulated over covisibility groups and pruned to the best ones. min_score is
    // LoopClosing's adaptive reference (lowest similarity among the keyframe's
    // covisibles), a calibration aid for backends with scene-dependent scores (the
    // late BoW backend); MegaLoc ignores it (globally calibrated cosine, fixed floor).
    virtual std::vector<Keyframe> detect_loop_candidates(const Keyframe& keyframe, float min_score) = 0;

    // Relocalization candidates for a (lost) frame; compute(frame) must have run.
    virtual std::vector<Keyframe> detect_relocalization_candidates(Frame& frame) = 0;

protected:
    FeatureType verification_feature_;
};

// `vpr: none`: inert backend, so the rest of the system needs no null checks — every
// query returns nothing, every compute is a no-op.
class PlaceRecognitionNone final : public PlaceRecognition
{
public:
    explicit PlaceRecognitionNone(const FeatureType verification_feature)
        : PlaceRecognition(verification_feature) {}
    std::string name() const override { return "none"; }
    bool is_active() const override { return false; }
    void compute(Frame&) override {}
    void compute(KeyFrame&) override {}
    void add(const Keyframe&) override {}
    void erase(const Keyframe&) override {}
    void clear() override {}
    float score(const KeyFrame&, const KeyFrame&) const override { return 0.0f; }
    std::vector<Keyframe> detect_loop_candidates(const Keyframe&, float) override { return {}; }
    std::vector<Keyframe> detect_relocalization_candidates(Frame&) override { return {}; }
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_PLACE_RECOGNITION_H
