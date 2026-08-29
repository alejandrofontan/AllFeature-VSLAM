/**
 * Module: AllFeature-VSLAM - PlaceRecognitionMegaLoc.cc
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 */

#include "PlaceRecognitionMegaLoc.h"

#include <algorithm>
#include <atomic>
#include <map>

#include "Frame.h"
#include "KeyFrame.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

PlaceRecognitionMegaLoc::PlaceRecognitionMegaLoc(const std::string& onnx_path, const std::string& precision,
                                                 const FeatureType verification_feature,
                                                 const PlaceRecognitionMegaLocParameters& params)
    : PlaceRecognition(verification_feature), params_(params)
{
    engine_ = std::make_unique<megaloc::TensorRTMegaLoc>(onnx_path, precision);
    // Absorb the one-time lazy CUDA/TRT warmup here rather than on the first keyframe.
    (void)engine_->infer(cv::Mat::zeros(480, 640, CV_8UC3));
}

std::vector<float> PlaceRecognitionMegaLoc::embed(const cv::Mat& image)
{
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return engine_->infer(image);
}

void PlaceRecognitionMegaLoc::compute(Frame& frame)
{
    if(!frame.global_descriptor.empty())
        return;
    if(frame.image.empty())
    {
        static std::atomic<bool> warned{false};
        if(!warned.exchange(true))
            AF_WARN("[PlaceRecognitionMegaLoc] frame " << frame.frame_id
                    << " has no image to embed (Frame::image empty) — relocalization query skipped");
        return;
    }
    frame.global_descriptor = embed(frame.image);
}

void PlaceRecognitionMegaLoc::compute(KeyFrame& keyframe)
{
    if(keyframe.global_descriptor.empty())
    {
        if(keyframe.image.empty())
        {
            static std::atomic<bool> warned{false};
            if(!warned.exchange(true))
                AF_WARN("[PlaceRecognitionMegaLoc] keyframe " << keyframe.keyId
                        << " has no image to embed (KeyFrame::image empty) — it will never be retrieved");
            return;
        }
        keyframe.global_descriptor = embed(keyframe.image);
    }
    // The image only existed for this purpose.
    keyframe.image.release();
}

void PlaceRecognitionMegaLoc::add(const Keyframe& keyframe)
{
    std::lock_guard<std::mutex> lock(database_mutex_);
    if(std::find(keyframes_.begin(), keyframes_.end(), keyframe) == keyframes_.end())
        keyframes_.push_back(keyframe);
}

void PlaceRecognitionMegaLoc::erase(const Keyframe& keyframe)
{
    std::lock_guard<std::mutex> lock(database_mutex_);
    keyframes_.erase(std::remove(keyframes_.begin(), keyframes_.end(), keyframe), keyframes_.end());
}

void PlaceRecognitionMegaLoc::clear()
{
    std::lock_guard<std::mutex> lock(database_mutex_);
    keyframes_.clear();
}

float PlaceRecognitionMegaLoc::score(const KeyFrame& a, const KeyFrame& b) const
{
    return megaloc::TensorRTMegaLoc::cosine(a.global_descriptor, b.global_descriptor);
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::retrieve(const std::vector<float>& query,
                                                        const std::set<KeyframeId>& excluded,
                                                        const float floor) const
{
    if(query.empty())
        return {};

    std::vector<Keyframe> database;
    {
        std::lock_guard<std::mutex> lock(database_mutex_);
        database = keyframes_;
    }

    // 1) Score every admissible keyframe; keep those at or above the floor
    //    (BoW analogue: "shares enough words" + score >= minScore).
    struct Scored { float similarity; Keyframe keyframe; };
    std::map<KeyframeId, Scored> scored;
    for(const Keyframe& candidate : database)
    {
        if(excluded.count(candidate->keyId) || candidate->is_bad() || candidate->global_descriptor.empty())
            continue;
        const float s = megaloc::TensorRTMegaLoc::cosine(query, candidate->global_descriptor);
        if(s >= floor)
            scored.emplace(candidate->keyId, Scored{s, candidate});
    }
    if(scored.empty())
        return {};

    // 2) Accumulate each candidate's score over its best covisible neighbours that are
    //    candidates themselves, and represent the group by its best-scoring member
    //    (same rule as KeyFrameDatabase::DetectLoopCandidates / detect_relocalization_candidates).
    struct Group { float accumulated; Keyframe best; };
    std::vector<Group> groups;
    groups.reserve(scored.size());
    float best_accumulated = 0.0f;
    for(const auto& [id, entry] : scored)
    {
        float accumulated = entry.similarity;
        float best_score = entry.similarity;
        Keyframe best = entry.keyframe;
        for(const Keyframe& neighbour : entry.keyframe->get_best_covisibility_keyframes(10))
        {
            const auto it = scored.find(neighbour->keyId);
            if(it == scored.end())
                continue;
            accumulated += it->second.similarity;
            if(it->second.similarity > best_score)
            {
                best_score = it->second.similarity;
                best = it->second.keyframe;
            }
        }
        groups.push_back(Group{accumulated, best});
        best_accumulated = std::max(best_accumulated, accumulated);
    }

    // 3) Keep the groups scoring above 0.75 x the best, best first, deduplicated by
    //    representative, capped at max_candidates.
    std::sort(groups.begin(), groups.end(),
              [](const Group& a, const Group& b) { return a.accumulated > b.accumulated; });
    const float min_to_retain = 0.75f * best_accumulated;
    std::set<KeyframeId> added;
    std::vector<Keyframe> candidates;
    for(const Group& group : groups)
    {
        if(group.accumulated <= min_to_retain)
            break;
        if(!added.insert(group.best->keyId).second)
            continue;
        candidates.push_back(group.best);
        if(params_.max_candidates > 0 && static_cast<int>(candidates.size()) >= params_.max_candidates)
            break;
    }
    return candidates;
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::detect_loop_candidates(const Keyframe& keyframe, const float min_score)
{
    std::set<KeyframeId> excluded{keyframe->keyId};
    for(const auto& [id, connected] : keyframe->GetConnectedKeyFrames())
        excluded.insert(id);
    // LoopClosing's min_score (lowest similarity to a covisible) can be arbitrarily low
    // when a covisible looks very different; the configured floor bounds it from below.
    return retrieve(keyframe->global_descriptor, excluded, std::max(min_score, params_.min_similarity));
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::detect_relocalization_candidates(Frame& frame)
{
    compute(frame);
    return retrieve(frame.global_descriptor, {}, params_.min_similarity);
}

} // namespace AF_VSLAM
