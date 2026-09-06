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
#include <cmath>
#include <limits>
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
    // Store in placecell under the keyframe's frame_id (idempotent): the descriptor the
    // frame already carries (embedded by keyframe_information on the tracking thread),
    // or embed the image now.
    if(!place_cell_->has(keyframe.frame_id))
    {
        if(keyframe.global_descriptor.size() != 0)
            place_cell_->add(keyframe.frame_id, keyframe.global_descriptor);
        else if(!keyframe.image.empty())
            place_cell_->add_image(keyframe.frame_id, keyframe.image);
        else
        {
            static std::atomic<bool> warned{false};
            if(!warned.exchange(true))
                AF_WARN("[PlaceRecognitionMegaLoc] keyframe " << keyframe.keyId
                        << " has no image to embed (KeyFrame::image empty) — it will never be retrieved");
            return;
        }
    }
    // The image and the descriptor copy only existed for this purpose (placecell holds
    // the stored descriptor: place_cell->descriptor(frame_id)).
    keyframe.image.release();
    keyframe.global_descriptor.resize(0);
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
    {
        std::lock_guard<std::mutex> lock(database_mutex_);
        keyframes_.clear();
    }
    // System reset: drop the stored descriptors and the VPR kernel too, so a fresh
    // map does not inherit rows from the wiped one.
    place_cell_->clear();
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
    //    (same rule classic ORB-SLAM2's keyframe database applied to BoW scores).
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

std::vector<Keyframe> PlaceRecognitionMegaLoc::detect_loop_candidates(const Keyframe& keyframe, const float /*min_score*/)
{
    std::set<KeyframeId> excluded{keyframe->keyId};
    for(const auto& [id, connected] : keyframe->GetConnectedKeyFrames())
        excluded.insert(id);
    // The fixed floor alone gates candidates. LoopClosing's adaptive reference score
    // (lowest similarity to a covisible keyframe) is deliberately IGNORED for this
    // backend: it exists to calibrate BoW's scene-dependent scores, while MegaLoc's
    // cosine is globally calibrated -- and with covisibles scoring ~0.75-0.9, the
    // adaptive bar would reject genuine revisits seen from a different viewpoint,
    // direction, or season (which legitimately score ~0.6-0.7).
    const Eigen::VectorXf* query = place_cell_->descriptor(keyframe->frame_id);
    if(!query)
        return {};
    return retrieve(*query, excluded, params_.min_similarity);
}

std::vector<Keyframe> PlaceRecognitionMegaLoc::detect_relocalization_candidates(Frame& frame)
{
    compute(frame);
    return retrieve(frame.global_descriptor, {}, params_.min_similarity);
}

std::optional<KeyframeInformation> PlaceRecognitionMegaLoc::keyframe_information(Frame& frame,
                                                                                 const std::vector<Keyframe>& window,
                                                                                 const bool centred)
{
    std::vector<placecell::PlaceCell::ExternalId> window_ids;
    window_ids.reserve(window.size());
    for(const Keyframe& keyframe : window)
        if(keyframe && !keyframe->is_bad())
            window_ids.push_back(keyframe->frame_id);

    // Read-only query on placecell: the frame's view is NOT stored. The embedding is
    // cached in the frame (reused by a relocalization query on the same frame, and by
    // compute(KeyFrame&) if the frame becomes a keyframe).
    placecell::PlaceCell::Information information;
    if(frame.global_descriptor.size() != 0)
        information = place_cell_->unexplained_information(frame.global_descriptor, &window_ids, centred);
    else if(!frame.image.empty())
        information = place_cell_->unexplained_information(frame.image, &window_ids, centred, &frame.global_descriptor);
    else
    {
        static std::atomic<bool> warned{false};
        if(!warned.exchange(true))
            AF_WARN("[PlaceRecognitionMegaLoc] frame " << frame.frame_id
                    << " has no image to embed (Frame::image empty) — keyframe information unavailable");
        return std::nullopt;
    }
    if(std::isnan(information.unexplained))
        return std::nullopt;

    KeyframeInformation result;
    result.unexplained = information.unexplained;
    result.explainers = information.explainers;
    result.best_explainer = FrameId(information.best_explainer);
    result.best_similarity = information.best_similarity;
    return result;
}

void PlaceRecognitionMegaLoc::record_keyframe_thresholds(const float tau, const float min_information)
{
    // The recorder keeps a change history and ignores repeated identical values, so
    // calling this every frame is cheap and the plots get a step wherever the Viewer
    // slider or a settings change moved a threshold.
    place_cell_->recorder().set_thresholds(tau, min_information);
}

void PlaceRecognitionMegaLoc::record_keyframe_decision(const FrameId frame_id, const bool inserted,
                                                       const std::optional<float> unexplained,
                                                       const std::string& reason)
{
    place_cell_->recorder().record_decision(frame_id, inserted,
                                            unexplained.value_or(std::numeric_limits<float>::quiet_NaN()), reason);
}

} // namespace AF_VSLAM
