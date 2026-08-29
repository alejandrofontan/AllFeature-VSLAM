/**
 * Module: AllFeature-VSLAM - PlaceRecognitionBoW.cc
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 */

#include "PlaceRecognitionBoW.h"

#include "Frame.h"
#include "KeyFrame.h"

namespace AF_VSLAM
{

PlaceRecognitionBoW::PlaceRecognitionBoW(std::shared_ptr<Vocabulary> vocabulary)
    : PlaceRecognition(vocabulary->featureType),
      vocabulary_(std::move(vocabulary)),
      database_(std::make_shared<KeyFrameDatabase>(vocabulary_))
{}

void PlaceRecognitionBoW::compute(Frame& frame)
{
    if(!vocabulary_->is_active())
        return;
    if(frame.mBowVec.empty())
        vocabulary_->transform(frame.descriptors.at(vocabulary_->featureType), frame.mBowVec, frame.mFeatVec);
}

void PlaceRecognitionBoW::compute(KeyFrame& keyframe)
{
    if(!vocabulary_->is_active())
        return;
    if(keyframe.mBowVec.empty() || keyframe.mFeatVec.empty())
    {
        // Feature vector associates features with nodes in the 4th level (from leaves up);
        // assumes a 6-level vocabulary tree (Vocabulary::transform hardcodes the 4).
        vocabulary_->transform(keyframe.descriptors.at(vocabulary_->featureType), keyframe.mBowVec, keyframe.mFeatVec);
    }
}

void PlaceRecognitionBoW::add(const Keyframe& keyframe) { database_->add(keyframe); }
void PlaceRecognitionBoW::erase(const Keyframe& keyframe) { database_->erase(keyframe); }
void PlaceRecognitionBoW::clear() { database_->clear(); }

float PlaceRecognitionBoW::score(const KeyFrame& a, const KeyFrame& b) const
{
    if(a.mBowVec.empty() || b.mBowVec.empty())
        return 0.0f;
    return static_cast<float>(vocabulary_->score(a.mBowVec, b.mBowVec));
}

std::vector<Keyframe> PlaceRecognitionBoW::detect_loop_candidates(const Keyframe& keyframe, const float min_score)
{
    return database_->DetectLoopCandidates(keyframe, min_score);
}

std::vector<Keyframe> PlaceRecognitionBoW::detect_relocalization_candidates(Frame& frame)
{
    return database_->detect_relocalization_candidates(&frame);
}

} // namespace AF_VSLAM
