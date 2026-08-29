/**
 * Module: AllFeature-VSLAM - PlaceRecognitionBoW.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 *
 * DBoW2 bag-of-words backend of the PlaceRecognition interface: a thin seam over the
 * pre-existing Vocabulary (descriptor -> BoW vector, score) and KeyFrameDatabase
 * (inverted file, candidate queries). Behaviour is identical to the code it wraps.
 */

#ifndef AF_VSLAM_PLACE_RECOGNITION_BOW_H
#define AF_VSLAM_PLACE_RECOGNITION_BOW_H

#include <memory>

#include "KeyFrameDatabase.h"
#include "PlaceRecognition.h"
#include "Vocabulary.h"

namespace AF_VSLAM
{

class PlaceRecognitionBoW final : public PlaceRecognition
{
public:
    // `vocabulary` must be created and loaded (System does that before constructing
    // the backend); its featureType doubles as the verification feature.
    explicit PlaceRecognitionBoW(std::shared_ptr<Vocabulary> vocabulary);

    std::string name() const override { return "bow"; }
    bool is_active() const override { return vocabulary_->is_active(); }

    void compute(Frame& frame) override;
    void compute(KeyFrame& keyframe) override;

    void add(const Keyframe& keyframe) override;
    void erase(const Keyframe& keyframe) override;
    void clear() override;

    float score(const KeyFrame& a, const KeyFrame& b) const override;

    std::vector<Keyframe> detect_loop_candidates(const Keyframe& keyframe, float min_score) override;
    std::vector<Keyframe> detect_relocalization_candidates(Frame& frame) override;

private:
    std::shared_ptr<Vocabulary> vocabulary_;
    std::shared_ptr<KeyFrameDatabase> database_;
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_PLACE_RECOGNITION_BOW_H
