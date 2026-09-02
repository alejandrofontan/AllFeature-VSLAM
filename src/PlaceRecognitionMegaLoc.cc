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

PlaceRecognitionMegaLoc::PlaceRecognitionMegaLoc(std::shared_ptr<placecell::MegaLocPlaceCell> place_cell,
                                                 const FeatureType verification_feature,
                                                 const PlaceRecognitionMegaLocParameters& params)
    : PlaceRecognition(verification_feature), params_(params), place_cell_(std::move(place_cell))
{
}

void PlaceRecognitionMegaLoc::compute(Frame& frame)
{
    // Relocalization QUERY: transient descriptor, deliberately NOT stored in placecell
    // (queries arrive every frame while tracking is lost; the store holds keyframes only).
    if(frame.global_descriptor.size() != 0)
        return;
    if(frame.image.empty())
    {
        static std::atomic<bool> warned{false};
        if(!warned.exchange(true))
            AF_WARN("[PlaceRecognitionMegaLoc] frame " << frame.frame_id
                    << " has no image to embed (Frame::image empty) — relocalization query skipped");
        return;
    }
    frame.global_descriptor = place_cell_->embedder().embed(frame.image);
}

void PlaceRecognitionMegaLoc::compute(KeyFrame& keyframe)
{
    // Embed + store in placecell under the keyframe's frame_id (idempotent)
    if(!place_cell_->has(keyframe.frame_id))
    {
        if(keyframe.image.empty())
        {
            static std::atomic<bool> warned{false};
            if(!warned.exchange(true))
                AF_WARN("[PlaceRecognitionMegaLoc] keyframe " << keyframe.keyId
                        << " has no image to embed (KeyFrame::image empty) — it will never be retrieved");
            return;
        }
        place_cell_->add_image(keyframe.frame_id, keyframe.image);
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
    const Eigen::VectorXf* da = place_cell_->descriptor(a.frame_id);
    const Eigen::VectorXf* db = place_cell_->descriptor(b.frame_id);
    if(!da || !db)
        return 0.0f;
    return placecell::MegaLocEmbedder::cosine(*da, *db);
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::retrieve(const Eigen::Ref<const Eigen::VectorXf>& query,
                                                        const std::set<KeyframeId>& excluded,
                                                        const float floor) const
{
    if(query.size() == 0)
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
        if(excluded.count(candidate->keyId) || candidate->is_bad())
            continue;
        const Eigen::VectorXf* descriptor = place_cell_->descriptor(candidate->frame_id);
        if(!descriptor)
            continue;
        const float s = placecell::MegaLocEmbedder::cosine(query, *descriptor);
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
    const Eigen::VectorXf* query = place_cell_->descriptor(keyframe->frame_id);
    if(!query)
        return {};
    return retrieve(*query, excluded, std::max(min_score, params_.min_similarity));
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::detect_relocalization_candidates(Frame& frame)
{
    compute(frame);
    return retrieve(frame.global_descriptor, {}, params_.min_similarity);
}

} // namespace AF_VSLAM
