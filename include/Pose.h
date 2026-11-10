#pragma once

#include "Types.h"

namespace AF_VSLAM
{

// A world->camera pose and every block derived from it.
//
// Naming convention, used throughout the codebase: Xab maps FROM frame b TO frame a.
// So Tcw takes world coordinates to camera coordinates, Rwc is the rotation from camera
// to world, and twc is the camera centre expressed in world coordinates. The accessors
// on Frame and KeyFrame are named after the block they return (get_Rcw, get_twc, ...)
// so the frame is visible at the call site rather than implied by a word like
// "translation", which fits tcw and twc equally well.
//
// Shared by Frame and KeyFrame: both cached the same information in different decompositions
// (Frame kept Tcw/Rcw/tcw/Rwc/twc, KeyFrame kept Tcw/Twc/twc) behind two set_pose
// implementations that disagreed on what to keep and on how to validate the input.
//
// The six members are a CACHE of Tcw, not six independent values: set() is the only writer and
// derives all of them in one pass, so they cannot drift apart.
//
// The lock stays OUTSIDE this struct. A Frame is thread-confined and needs none; a KeyFrame is
// shared and guards its Pose with pose_mutex_.
struct Pose
{
    mat4f Tcw{mat4f::Identity()};
    mat4f Twc{mat4f::Identity()};
    mat3f Rcw{mat3f::Identity()};
    mat3f Rwc{mat3f::Identity()};
    vec3f tcw{vec3f::Zero()};
    vec3f twc{vec3f::Zero()};

    // False, and the pose left untouched, when Tcw_ is not finite: a diverged optimisation
    // produces NaN, and keeping the last good pose beats poisoning every downstream
    // projection with it.
    //
    // Rwc is Rcw.transpose(), which is the inverse only while Rcw is orthonormal. Every writer
    // passes a rotation that came from a quaternion (Converter::to_matrix4f on a g2o SE3Quat
    // or Sim3), so it holds to float precision. This does NOT re-orthonormalise; a caller that
    // assembles Tcw by hand has to keep that invariant itself.
    bool set(const mat4f& Tcw_);
};

}
