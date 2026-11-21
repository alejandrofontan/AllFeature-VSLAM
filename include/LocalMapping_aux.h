#pragma once

#include <chrono>
#include <map>

namespace AF_VSLAM {

struct LocalMappingProfiler
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start{Clock::now()};

    void iteration_done(std::map<int, int>& histogram) const {
        histogram[int(std::chrono::duration<double, std::milli>(Clock::now() - t_start).count())]++;
    }
};

}

