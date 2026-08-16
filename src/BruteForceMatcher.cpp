#include "BruteForceMatcher.h"

#include <Eigen/Core>

#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace AF_VSLAM
{

namespace
{

// Shared cross-check skeleton: one full distance sweep tracks the best train index per
// query (thread-owned) and the best query index per train (thread-local, merged after),
// then keeps only mutual pairs. Tie-breaking must match OpenCV's batchDistance argmin
// scans: strictly-smaller wins, so the FIRST minimal index survives in both directions;
// the train-side merge therefore prefers (smaller distance, then smaller query index).
template <typename DistT, typename RowDistFn>
std::vector<cv::DMatch> crosscheck_match(const int nq, const int nt, const DistT distMax,
                                         const RowDistFn& rowDist)
{
    std::vector<cv::DMatch> matches;
    if (nq == 0 || nt == 0)
        return matches;

    std::vector<DistT> qBest(nq, distMax);
    std::vector<int> qBestIdx(nq, -1);
    std::vector<DistT> tBest(nt, distMax);
    std::vector<int> tBestQ(nt, INT_MAX);

    #pragma omp parallel
    {
        std::vector<DistT> tBestLocal(nt, distMax);
        std::vector<int> tBestQLocal(nt, INT_MAX);

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < nq; i++)
        {
            DistT best = distMax;
            int bestIdx = -1;
            for (int j = 0; j < nt; j++)
            {
                const DistT d = rowDist(i, j);
                if (d < best)
                {
                    best = d;
                    bestIdx = j;
                }
                if (d < tBestLocal[j])   // scan order over i is ascending within a thread,
                {                        // so first-minimum per train is preserved locally
                    tBestLocal[j] = d;
                    tBestQLocal[j] = i;
                }
            }
            qBest[i] = best;
            qBestIdx[i] = bestIdx;
        }

        #pragma omp critical
        for (int j = 0; j < nt; j++)
        {
            if (tBestLocal[j] < tBest[j]
                || (tBestLocal[j] == tBest[j] && tBestQLocal[j] < tBestQ[j]))
            {
                tBest[j] = tBestLocal[j];
                tBestQ[j] = tBestQLocal[j];
            }
        }
    }

    matches.reserve(nq);
    for (int i = 0; i < nq; i++)
    {
        const int j = qBestIdx[i];
        if (j >= 0 && tBestQ[j] == i)
            matches.emplace_back(i, j, static_cast<float>(qBest[i]));
    }
    return matches;
}

} // namespace

std::vector<cv::DMatch> bruteforce_match_hamming(const cv::Mat& query, const cv::Mat& train)
{
    CV_Assert(query.empty() || query.type() == CV_8U);
    CV_Assert(train.empty() || train.type() == CV_8U);
    CV_Assert(query.empty() || train.empty() || query.cols == train.cols);

    const int nbytes = query.cols;
    const int nwords = nbytes / 8;

    const auto rowDist = [&](int i, int j) -> int {
        const uint8_t* q = query.ptr<uint8_t>(i);
        const uint8_t* t = train.ptr<uint8_t>(j);
        int d = 0;
        for (int w = 0; w < nwords; w++)
        {
            uint64_t a, b;   // memcpy: descriptor rows have no 8-byte alignment guarantee
            std::memcpy(&a, q + 8 * w, 8);
            std::memcpy(&b, t + 8 * w, 8);
            d += __builtin_popcountll(a ^ b);
        }
        for (int r = nwords * 8; r < nbytes; r++)   // tail for widths not divisible by 8 (akaze61)
            d += __builtin_popcount(static_cast<unsigned>(q[r] ^ t[r]));
        return d;
    };

    return crosscheck_match<int>(query.rows, train.rows, INT_MAX, rowDist);
}

std::vector<cv::DMatch> bruteforce_match_l2(const cv::Mat& query, const cv::Mat& train)
{
    CV_Assert(query.empty() || query.type() == CV_32F);
    CV_Assert(train.empty() || train.type() == CV_32F);
    CV_Assert(query.empty() || train.empty() || query.cols == train.cols);

    const int dim = query.cols;
    using VecMap = Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned>;

    // Selection runs on SQUARED distances (monotone under sqrt, so the argmin — and
    // OpenCV's ordering — is unchanged); the reported DMatch distance is the true L2.
    const auto rowDist = [&](int i, int j) -> float {
        return (VecMap(query.ptr<float>(i), dim) - VecMap(train.ptr<float>(j), dim)).squaredNorm();
    };

    std::vector<cv::DMatch> matches =
        crosscheck_match<float>(query.rows, train.rows, std::numeric_limits<float>::max(), rowDist);
    for (auto& m : matches)
        m.distance = std::sqrt(m.distance);
    return matches;
}

} // namespace AF_VSLAM
