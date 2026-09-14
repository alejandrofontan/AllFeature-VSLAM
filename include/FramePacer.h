/**
 * @file FramePacer.h
 * @brief Real-time replay pacing for the offline CLI entry points, with a configurable
 *        upper bound on the pause between consecutive frames.
 *
 * The mono and rgbd executables replay a recorded sequence at its own rate: after
 * tracking frame i they sleep for the remainder of the timestamp gap to frame i+1.
 * A gap far larger than the nominal frame period (dropped frames, holes in the
 * recording, a paused capture) would otherwise stall the run for the whole gap, so the
 * pause is capped at `max_frame_wait` seconds (settings yaml key of the same name).
 * A missing, zero or negative key keeps the historical uncapped behaviour.
 *
 * @author Alejandro Fontan
 * @version 1.0
 * @date Created: 2026-09-15
 * @date Updated: 2026-09-15
 */

#ifndef AF_VSLAM_FRAME_PACER_H
#define AF_VSLAM_FRAME_PACER_H

#include <chrono>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "Types.h"

namespace AF_VSLAM {

class FramePacer {
public:
    /// Settings yaml key read by `load()`.
    static constexpr const char* kSettingsKey = "max_frame_wait";

    /// Reads `max_frame_wait` (seconds) from the loaded settings; absent or <= 0 disables the cap.
    void load(const YAML::Node& settings) {
        maxWait_ = 0.0;
        if (settings[kSettingsKey])
            maxWait_ = settings[kSettingsKey].as<Seconds>();
        if (maxWait_ < 0.0)
            maxWait_ = 0.0;
    }

    /// True when a positive cap is configured.
    bool enabled() const { return maxWait_ > 0.0; }

    /// Configured cap in seconds (0 = uncapped).
    Seconds maxWait() const { return maxWait_; }

    /**
     * Sleeps for the remainder of the timestamp gap `T` after a frame that took `ttrack`
     * seconds to process, i.e. `T - ttrack`, capped at `max_frame_wait` when enabled.
     * Does nothing when tracking already used up the gap.
     */
    void wait(const Seconds T, const Seconds ttrack) {
        const Seconds remaining = T - ttrack;
        if (remaining <= 0.0)
            return;

        Seconds sleep = remaining;
        if (enabled() && sleep > maxWait_) {
            sleep = maxWait_;
            ++nCapped_;
            timeSaved_ += remaining - sleep;
        }
        std::this_thread::sleep_for(std::chrono::duration<Seconds>(sleep));
    }

    /// Number of waits shortened by the cap so far.
    size_t numCapped() const { return nCapped_; }

    /// Total sleep time removed by the cap so far, in seconds.
    Seconds timeSaved() const { return timeSaved_; }

    /// One-line end-of-run summary (only meaningful when the cap is enabled).
    void printSummary(std::ostream& os = std::cout) const {
        if (!enabled())
            return;
        os << "frame waits capped at " << maxWait_ << " s: " << nCapped_
           << " (saved " << timeSaved_ << " s)" << std::endl;
    }

private:
    Seconds maxWait_{0.0};
    size_t nCapped_{0};
    Seconds timeSaved_{0.0};
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_FRAME_PACER_H
