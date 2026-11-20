#include "Calibration.h"
#include "afvslam_log.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace AF_VSLAM
{
namespace
{
// Four corners only bound the undistorted image when the radial map is monotonically
// EXPANDING: along an edge the offset from the principal point is constant and only the
// radius varies, so the extremum sits at the largest radius — the corner. Under pincushion
// the map shrinks with radius, putting the extremum at the EDGE MIDPOINT instead, and
// tangential terms (p1, p2) break radial symmetry altogether. Sampling the whole border
// assumes neither. t spans [0,1] inclusive, so the corners are still among the samples and
// the resulting box is always a superset of the corner-only one.
constexpr int border_samples_per_edge = 32;

constexpr float round_trip_tolerance_px = 1.0f;

// cv::undistortPoints inverts the distortion with a fixed-point iteration that does not always
// converge; OpenCV's divergence guard then returns the INPUT unchanged, which no inspection of
// the output can distinguish from a correct answer. Re-distorting each sample can. Across the
// benchmark's 13 distortion models only ENDOMAPPER (f≈319 across 640px, i.e. >90 deg FOV through
// a 4-coefficient radtan model) fails — and it fails on 90 of 128 samples, not marginally. So a
// single failure is read as "this model cannot be inverted here", not as one bad sample.
bool border_round_trips(const std::vector<cv::Point2f>& original,
                        const std::vector<cv::Point2f>& undistorted,
                        const cv::Mat& K, const cv::Mat& distortion_coefficients)
{
    const float cx = K.at<float>(0,2), cy = K.at<float>(1,2);
    const float inv_fx = 1.0f / K.at<float>(0,0), inv_fy = 1.0f / K.at<float>(1,1);

    std::vector<cv::Point3f> normalized;
    normalized.reserve(undistorted.size());
    for(const cv::Point2f& p : undistorted)
        normalized.emplace_back((p.x - cx) * inv_fx, (p.y - cy) * inv_fy, 1.0f);

    std::vector<cv::Point2f> back;
    cv::projectPoints(normalized, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0), K, distortion_coefficients, back);

    for(std::size_t i = 0; i < original.size(); ++i)
        if(cv::norm(back[i] - original[i]) > round_trip_tolerance_px)
            return false;
    return true;
}
}

bool Calibration::has_distortion() const
{
    // Every coefficient, not just k1: a model can have k1 == 0 with k2/k3 or the tangential
    // p1/p2 non-zero, and testing only the first would skip undistortion entirely.
    return cv::countNonZero(distortion_coefficients) > 0;
}

std::vector<cv::Point2f> Calibration::undistort(const std::vector<cv::Point2f>& points) const
{
    if(!has_distortion() || points.empty())
        return points;

    std::vector<cv::Point2f> out(points.size());
    cv::undistortPoints(points, out, K, distortion_coefficients, cv::noArray(), K);
    return out;
}

Calibration::Calibration(const cv::Mat& K_, const cv::Mat& distortion_coefficients_,
                         int width_, int height_, int grid_cols, int grid_rows,
                         float depth_noise_coeff_):
    K(K_.clone()), distortion_coefficients(distortion_coefficients_.clone()),
    width(width_), height(height_),
    fx(K.at<float>(0,0)), fy(K.at<float>(1,1)),
    cx(K.at<float>(0,2)), cy(K.at<float>(1,2)),
    invfx(1.0f / fx), invfy(1.0f / fy),
    depth_noise_coeff(depth_noise_coeff_)
{
    const float w = static_cast<float>(width);
    const float h = static_cast<float>(height);

    std::vector<cv::Point2f> border;
    border.reserve(4 * border_samples_per_edge);
    for(int i = 0; i < border_samples_per_edge; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(border_samples_per_edge - 1);
        border.emplace_back(t * w, 0.0f);   // top
        border.emplace_back(t * w, h);      // bottom
        border.emplace_back(0.0f,  t * h);  // left
        border.emplace_back(w,     t * h);  // right
    }

    const std::vector<cv::Point2f> undistorted = undistort(border);

    if(has_distortion() && !border_round_trips(border, undistorted, K, distortion_coefficients))
    {
        AF_WARN("[Calibration] the distortion model could not be inverted on the image border"
                << " (cv::undistortPoints did not converge) — falling back to the raw "
                << width << "x" << height << " rectangle for the undistorted bounds."
                << " Keypoint undistortion is unreliable for this camera.");
        min_x = 0.0f; max_x = w;
        min_y = 0.0f; max_y = h;
    }
    else
    {
        min_x = max_x = undistorted[0].x;
        min_y = max_y = undistorted[0].y;
        for(const cv::Point2f& p : undistorted)
        {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
    }

    grid_element_width_inv  = static_cast<float>(grid_cols) / (max_x - min_x);
    grid_element_height_inv = static_cast<float>(grid_rows) / (max_y - min_y);
}

}
