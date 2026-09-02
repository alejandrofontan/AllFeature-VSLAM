/**
 * Module: AllFeature-VSLAM - PlaceRecognitionMegaLoc.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 *
 * MegaLoc backend of the PlaceRecognition interface (issue #17, stage 3): one 8448-d
 * L2-normalised global descriptor per keyframe, computed AND STORED by placecell
 * (Thirdparty/placecell MegaLocPlaceCell, owned by System and injected here), keyed
 * by the keyframe's frame_id — keyframes carry no descriptor member. Relocalization
 * query frames are transient: embedded via the store's embedder() but never stored
 * (Frame::global_descriptor holds that query). Retrieval is brute-force cosine
 * similarity over the keyframe database, followed by the same covisibility
 * accumulation and 0.75-of-best pruning KeyFrameDatabase applies to BoW scores, so
 * the consumers (Tracking::relocalize, LoopClosing::DetectLoop) see the same
 * candidate semantics with a different similarity.
 *
 * Threading: compute() is called from the LocalMapping thread (keyframes) and the
 * tracking thread (relocalization query); placecell serialises internally. The
 * database has its own mutex; queries take a snapshot and score outside the lock.
 */

#ifndef AF_VSLAM_PLACE_RECOGNITION_MEGALOC_H
#define AF_VSLAM_PLACE_RECOGNITION_MEGALOC_H

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <placecell/megaloc_placecell.h>

#include "PlaceRecognition.h"

namespace AF_VSLAM
{

// Tunables, loaded from the settings YAML (all optional; compiled-in defaults otherwise).
struct PlaceRecognitionMegaLocParameters
{
    // Cosine floor for a keyframe to become a candidate at all. MegaLoc's cosine has a
    // common-mode floor (~0.35-0.4 between unrelated places); consecutive frames of a
    // sequence score ~0.9, keyframe-spaced neighbours ~0.75.
    float min_similarity{0.55f};
    // Cap on the candidates returned per query — each one costs a brute-force feature
    // match + PnP/Sim3 downstream (0 = unlimited).
    int max_candidates{10};
};

class PlaceRecognitionMegaLoc final : public PlaceRecognition
{
public:
    static constexpr const char* kDefaultOnnx = "megaloc_models/megaloc_322x322.onnx";

    // The placecell store (embedder + keyframe descriptor storage; engine build/load
    // and warmup already absorbed by its constructor) is owned by System and
    // injected here.
    PlaceRecognitionMegaLoc(std::shared_ptr<placecell::MegaLocPlaceCell> place_cell,
                            FeatureType verification_feature,
                            const PlaceRecognitionMegaLocParameters& params);

    std::string name() const override { return "megaloc"; }
    bool is_active() const override { return true; }
    bool needs_image() const override { return true; }

    void compute(Frame& frame) override;
    void compute(KeyFrame& keyframe) override;

    void add(const Keyframe& keyframe) override;
    void erase(const Keyframe& keyframe) override;
    void clear() override;

    float score(const KeyFrame& a, const KeyFrame& b) const override;

    std::vector<Keyframe> detect_loop_candidates(const Keyframe& keyframe, float min_score) override;
    std::vector<Keyframe> detect_relocalization_candidates(Frame& frame) override;

    const PlaceRecognitionMegaLocParameters& parameters() const { return params_; }
    const placecell::MegaLocPlaceCell& place_cell() const { return *place_cell_; }

private:
    // Shared retrieval: score every database keyframe (not excluded, not bad, with a
    // descriptor in placecell) against `query`, keep those >= floor, accumulate over
    // covisibility groups, return the best ones (see the .cc for the exact rule).
    std::vector<Keyframe> retrieve(const Eigen::Ref<const Eigen::VectorXf>& query,
                                   const std::set<KeyframeId>& excluded, float floor) const;

    PlaceRecognitionMegaLocParameters params_;

    std::shared_ptr<placecell::MegaLocPlaceCell> place_cell_;

    std::vector<Keyframe> keyframes_;
    mutable std::mutex database_mutex_;
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_PLACE_RECOGNITION_MEGALOC_H
