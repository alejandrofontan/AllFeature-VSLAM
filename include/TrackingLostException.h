#ifndef AF_VSLAM_TRACKING_LOST_EXCEPTION_H
#define AF_VSLAM_TRACKING_LOST_EXCEPTION_H

#include <stdexcept>
#include <string>

namespace AF_VSLAM
{

// Thrown by Tracking's per-frame tracking stages (TrackReferenceKeyFrame, TrackLocalMap) at the
// exact point a tracking-quality check fails, carrying a human-readable reason plus the statistics
// that triggered it. Caught in Tracking::Track(), which prints it and folds it into the state_=LOST
// transition — kept as its own type (rather than a bare std::runtime_error) so that catch sites only
// swallow tracking-loss signals, not unrelated errors from the same call stack.
class TrackingLostException : public std::runtime_error
{
public:
    explicit TrackingLostException(const std::string& reason) : std::runtime_error(reason) {}
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_TRACKING_LOST_EXCEPTION_H
