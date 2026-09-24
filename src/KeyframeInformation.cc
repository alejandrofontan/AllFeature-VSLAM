/**
 * Module: AllFeature-VSLAM - KeyframeInformation.cc
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-24
 * - License: GPLv3 License
 *
 * See KeyframeInformation.h. Both backends translate between the SLAM objects (Frame,
 * KeyFrame, MapPoint) and placecell's external ids (frame_id) / item ids (ptId); the
 * information maths lives in placecell.
 */

#include "KeyframeInformation.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#include <placecell/megaloc_placecell.h>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

namespace
{
// Alive keyframes of the window as placecell external ids
std::vector<placecell::PlaceCell::ExternalId> window_ids(const std::vector<Keyframe>& window)
{
    std::vector<placecell::PlaceCell::ExternalId> ids;
    ids.reserve(window.size());
    for(const Keyframe& keyframe : window)
        if(keyframe && !keyframe->is_bad())
            ids.push_back(keyframe->frame_id);
    return ids;
}

std::optional<KeyframeInformationValue> to_value(const placecell::PlaceCell::Information& information)
{
    if(std::isnan(information.unexplained))
        return std::nullopt;
    KeyframeInformationValue value;
    value.unexplained = information.unexplained;
    value.explainers = information.explainers;
    value.best_explainer = FrameId(information.best_explainer);
    value.best_similarity = information.best_similarity;
    return value;
}
} // namespace

// ---- Recorder feed ----------------------------------------------------------------------

void KeyframeInformation::record_thresholds(const float tau, const float min_information)
{
    // The recorder keeps a change history and ignores repeated identical values, so
    // calling this every frame is cheap and the plots get a step wherever the Viewer
    // slider or a settings change moved a threshold.
    place_cell().recorder().set_thresholds(tau, min_information);
}

void KeyframeInformation::record_decision(const FrameId frame_id, const bool inserted,
                                          const std::optional<float> unexplained, const std::string& reason)
{
    place_cell().recorder().record_decision(frame_id, inserted,
                                            unexplained.value_or(std::numeric_limits<float>::quiet_NaN()), reason);
}

// ---- megaloc ----------------------------------------------------------------------------

KeyframeInformationMegaLoc::KeyframeInformationMegaLoc(std::shared_ptr<placecell::MegaLocPlaceCell> place_cell)
    : place_cell_(std::move(place_cell))
{
}

KeyframeInformationMegaLoc::~KeyframeInformationMegaLoc() = default;

placecell::PlaceCell& KeyframeInformationMegaLoc::place_cell() { return *place_cell_; }
const placecell::PlaceCell& KeyframeInformationMegaLoc::place_cell() const { return *place_cell_; }

std::optional<KeyframeInformationValue> KeyframeInformationMegaLoc::information(Frame& frame,
                                                                                const std::vector<Keyframe>& window,
                                                                                const bool centred)
{
    const std::vector<placecell::PlaceCell::ExternalId> ids = window_ids(window);

    // Read-only query on placecell: the frame's view is NOT stored. The embedding is
    // cached in the frame (reused by a relocalization query on the same frame, and by
    // PlaceRecognitionMegaLoc::compute(KeyFrame&) if the frame becomes a keyframe).
    placecell::PlaceCell::Information information;
    if(frame.global_descriptor.size() != 0)
        information = place_cell_->unexplained_information(frame.global_descriptor, &ids, centred);
    else if(!frame.image.empty())
        information = place_cell_->unexplained_information(frame.image, &ids, centred, &frame.global_descriptor);
    else
    {
        static std::atomic<bool> warned{false};
        if(!warned.exchange(true))
            AF_WARN("[KeyframeInformation] frame " << frame.frame_id
                    << " has no image to embed (Frame::image empty) — keyframe information unavailable");
        return std::nullopt;
    }
    return to_value(information);
}

// ---- covisibility -----------------------------------------------------------------------

KeyframeInformationCovisibility::KeyframeInformationCovisibility(const placecell::PlaceCell::Options& options)
    : place_cell_(std::make_shared<placecell::PlaceCell>(options))
{
}

std::vector<placecell::PlaceCell::ItemId> KeyframeInformationCovisibility::observed_point_ids(const KeyFrame& keyframe)
{
    std::vector<placecell::PlaceCell::ItemId> items;
    for(const FeatureType feature_type : keyframe.featureTypes)
    {
        // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
        const std::vector<Pt> map_points = keyframe.get_map_point_matches(feature_type);
        items.reserve(items.size() + map_points.size());
        for(const Pt& map_point : map_points)
            if(map_point && !map_point->is_bad())
                items.push_back(placecell::PlaceCell::ItemId(map_point->ptId));
    }
    return items;   // placecell sorts and de-duplicates
}

std::optional<KeyframeInformationValue> KeyframeInformationCovisibility::information(Frame& frame,
                                                                                     const std::vector<Keyframe>& window,
                                                                                     const bool centred)
{
    // The view's item set: the map points the frame tracks after track_local_map —
    // non-null, non-bad, not flagged outlier by the pose optimisation (the same points
    // Tracking::update_local_keyframes votes with).
    std::vector<placecell::PlaceCell::ItemId> items;
    for(const auto& [feature_type, map_points] : frame.pts)
    {
        const auto outliers = frame.outliers.find(feature_type);
        for(size_t i = 0; i < map_points.size(); i++)
        {
            const Pt& map_point = map_points[i];
            if(!map_point || map_point->is_bad())
                continue;
            if(outliers != frame.outliers.end() && i < outliers->second.size() && outliers->second[i])
                continue;
            items.push_back(placecell::PlaceCell::ItemId(map_point->ptId));
        }
    }
    if(items.empty())
        return std::nullopt;   // nothing tracked: no cosine (the tracking triggers decide)

    const std::vector<placecell::PlaceCell::ExternalId> ids = window_ids(window);
    return to_value(place_cell_->unexplained_information(items, &ids, centred));
}

void KeyframeInformationCovisibility::on_keyframe_processed(const Keyframe& keyframe)
{
    if(!keyframe || keyframe->is_bad())
        return;
    // First call for this frame_id: the store appends the row (item mode from the first
    // keyframe on). The set is the keyframe's observations right after process_new_keyframe
    // (Tracking's matches); create_new_map_points / fuse extend it, refresh() picks that up.
    place_cell_->set_items(keyframe->frame_id, observed_point_ids(*keyframe));
}

void KeyframeInformationCovisibility::refresh(const std::vector<Keyframe>& alive)
{
    // Re-send every alive keyframe's current set; placecell diffs against what it holds
    // (an unchanged set costs one vector compare) and ignores culled rows (frozen).
    for(const Keyframe& keyframe : alive)
        if(keyframe && !keyframe->is_bad() && !place_cell_->is_culled(keyframe->frame_id))
            place_cell_->set_items(keyframe->frame_id, observed_point_ids(*keyframe));
}

void KeyframeInformationCovisibility::clear()
{
    place_cell_->clear();
}

} // namespace AF_VSLAM
