/**
 * test_bfmatcher_parity — parity + benchmark harness for BruteForceMatcher (issue #13 P2).
 *
 * Asserts that bruteforce_match_hamming/_l2 reproduce cv::BFMatcher(<norm>, crossCheck=true)
 * ::match() on random descriptor sets across shapes and widths, then benchmarks both on the
 * production-sized case. Hamming must be bit-exact (index AND distance); L2 must agree on
 * indices with distances equal to float tolerance (Eigen vs OpenCV accumulation order).
 * Exit code 0 = all parity checks passed.
 */

#include "BruteForceMatcher.h"
#include "FrameDrawer.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

int failures = 0;

std::vector<cv::DMatch> cvMatch(const cv::Mat& q, const cv::Mat& t, int norm)
{
    std::vector<cv::DMatch> m;
    if (q.empty() || t.empty())
        return m;
    cv::BFMatcher matcher(norm, /*crossCheck=*/true);
    matcher.match(q, t, m);
    return m;
}

void check(const std::string& name, const std::vector<cv::DMatch>& ref,
           const std::vector<cv::DMatch>& got, bool exactDistance)
{
    if (ref.size() != got.size())
    {
        std::printf("FAIL %-28s size %zu (cv) vs %zu (ours)\n", name.c_str(), ref.size(), got.size());
        failures++;
        return;
    }
    for (size_t k = 0; k < ref.size(); k++)
    {
        const auto& a = ref[k];
        const auto& b = got[k];
        const float dtol = exactDistance ? 0.0f : 1e-3f * std::max(1.0f, a.distance);
        if (a.queryIdx != b.queryIdx || a.trainIdx != b.trainIdx
            || std::abs(a.distance - b.distance) > dtol)
        {
            std::printf("FAIL %-28s match %zu: cv(q=%d,t=%d,d=%.6f) vs ours(q=%d,t=%d,d=%.6f)\n",
                        name.c_str(), k, a.queryIdx, a.trainIdx, a.distance,
                        b.queryIdx, b.trainIdx, b.distance);
            failures++;
            return;
        }
    }
    std::printf("PASS %-28s (%zu matches)\n", name.c_str(), ref.size());
}

void parityHamming(cv::RNG& rng, int nq, int nt, int cols)
{
    cv::Mat q(nq, cols, CV_8U), t(nt, cols, CV_8U);
    rng.fill(q, cv::RNG::UNIFORM, 0, 256);
    rng.fill(t, cv::RNG::UNIFORM, 0, 256);
    check("hamming " + std::to_string(nq) + "x" + std::to_string(nt) + "x" + std::to_string(cols),
          cvMatch(q, t, cv::NORM_HAMMING), AF_VSLAM::bruteforce_match_hamming(q, t),
          /*exactDistance=*/true);
}

void parityL2(cv::RNG& rng, int nq, int nt, int cols)
{
    cv::Mat q(nq, cols, CV_32F), t(nt, cols, CV_32F);
    rng.fill(q, cv::RNG::UNIFORM, -1.0, 1.0);
    rng.fill(t, cv::RNG::UNIFORM, -1.0, 1.0);
    check("l2 " + std::to_string(nq) + "x" + std::to_string(nt) + "x" + std::to_string(cols),
          cvMatch(q, t, cv::NORM_L2), AF_VSLAM::bruteforce_match_l2(q, t),
          /*exactDistance=*/false);
}

template <typename Fn>
double bestMs(const Fn& fn, int reps)
{
    double best = 1e30;
    for (int r = 0; r < reps; r++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

} // namespace

int main()
{
    // Diagnostic (P2 step 0): what parallel backend is this OpenCV build actually using?
    {
        const std::string info = cv::getBuildInformation();
        const size_t p = info.find("Parallel framework");
        const std::string line =
            p == std::string::npos ? "Parallel framework: (not reported)"
                                   : info.substr(p, info.find('\n', p) - p);
        std::printf("OpenCV %s | getNumThreads=%d | %s\n",
                    CV_VERSION, cv::getNumThreads(), line.c_str());
    }

    cv::RNG rng(0x5eed);

    // Empty-input behaviour: ours must return empty without touching cv.
    if (!AF_VSLAM::bruteforce_match_hamming(cv::Mat(), cv::Mat(0, 32, CV_8U)).empty()
        || !AF_VSLAM::bruteforce_match_l2(cv::Mat(0, 128, CV_32F), cv::Mat()).empty())
    {
        std::printf("FAIL empty-input\n");
        failures++;
    }
    else
        std::printf("PASS empty-input\n");

    for (const int cols : {32, 48, 61})   // orb32, brisk48, akaze61 (non-multiple-of-8 tail)
    {
        parityHamming(rng, 1, 1, cols);
        parityHamming(rng, 37, 53, cols);
        parityHamming(rng, 500, 700, cols);
        parityHamming(rng, 2000, 5400, cols);
    }
    for (const int cols : {64, 128, 256})   // surf64/kaze64, sift/aliked128, superpoint256
    {
        parityL2(rng, 1, 1, cols);
        parityL2(rng, 37, 53, cols);
        parityL2(rng, 500, 700, cols);
        parityL2(rng, 2000, 5400, cols);
    }

    // Benchmark at production scale (frame ~2000 x keyframe ~5400 descriptors).
    {
        cv::Mat q(2000, 32, CV_8U), t(5400, 32, CV_8U);
        rng.fill(q, cv::RNG::UNIFORM, 0, 256);
        rng.fill(t, cv::RNG::UNIFORM, 0, 256);
        std::printf("bench hamming 2000x5400x32: cv=%.1f ms  ours=%.1f ms\n",
                    bestMs([&] { cvMatch(q, t, cv::NORM_HAMMING); }, 5),
                    bestMs([&] { AF_VSLAM::bruteforce_match_hamming(q, t); }, 5));
    }
    {
        cv::Mat q(2000, 128, CV_32F), t(5400, 128, CV_32F);
        rng.fill(q, cv::RNG::UNIFORM, -1.0, 1.0);
        rng.fill(t, cv::RNG::UNIFORM, -1.0, 1.0);
        std::printf("bench l2 2000x5400x128: cv=%.1f ms  ours=%.1f ms\n",
                    bestMs([&] { cvMatch(q, t, cv::NORM_L2); }, 5),
                    bestMs([&] { AF_VSLAM::bruteforce_match_l2(q, t); }, 5));
    }

    std::printf(failures == 0 ? "ALL PARITY CHECKS PASSED\n"
                              : "PARITY FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
