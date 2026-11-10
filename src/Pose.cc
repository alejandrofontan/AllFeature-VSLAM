#include "Pose.h"

namespace AF_VSLAM
{

bool Pose::set(const mat4f& Tcw_)
{
    if(!Tcw_.allFinite())
        return false;

    Rcw = Tcw_.block<3,3>(0,0);
    tcw = Tcw_.block<3,1>(0,3);
    Rwc = Rcw.transpose();
    twc = -Rwc * tcw;

    // Both rebuilt from the blocks rather than copied: the input's bottom row is ignored, so
    // Tcw and Twc are well-formed even if the caller handed over a malformed 4x4. KeyFrame
    // used to copy Tcw verbatim while building Twc from Identity, so the two could disagree.
    Tcw = mat4f::Identity();
    Tcw.block<3,3>(0,0) = Rcw;
    Tcw.block<3,1>(0,3) = tcw;

    Twc = mat4f::Identity();
    Twc.block<3,3>(0,0) = Rwc;
    Twc.block<3,1>(0,3) = twc;
    return true;
}

}
