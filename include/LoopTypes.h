/**
 * Module: AllFeature-VSLAM - LoopTypes.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - Updated: 2026-11-21
 * - License: GPLv3 License
 *
 * The two types Loop Closing hands to the essential-graph optimization, so that Optimizer.h does not need
 * LoopClosing.h.
 */

#pragma once

#include <functional>
#include <map>
#include <utility>

#include <Eigen/Core>
#include "g2o/types/types_seven_dof_expmap.h"

#include "KeyFrame.h"
#include "Types.h"

namespace AF_VSLAM
{

// Keyframe -> Sim3 pose; aligned allocator for g2o::Sim3's fixed-size Eigen members
using KeyframePoses = std::map<Keyframe, g2o::Sim3, std::less<Keyframe>,
                               Eigen::aligned_allocator<std::pair<const Keyframe, g2o::Sim3>>>;

// A keyframe of the corrected side of a loop and the covisibility links it gained
// through the loop fusion: the loop edges of Optimizer::optimize_essential_graph.
struct LoopConnections
{
    Keyframe keyframe{};
    std::map<KeyframeId, Keyframe> connections{};
};

} // namespace AF_VSLAM
