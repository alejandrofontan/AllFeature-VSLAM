#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace AF_VSLAM
{

struct Calibration
{

    Calibration(const cv::Mat& K_, const cv::Mat& distortion_coefficients_,
                int width_, int height_, int grid_cols, int grid_rows,
                float depth_noise_coeff_);


    // The image-bounds test, half-open [min, max): max_x is the image WIDTH in the
    // undistorted case (see the constructor), so a coordinate of exactly max_x is outside a
    // [0, w) pixel grid. One definition, used by project_if_visible and by the matcher --
    // KeyFrame and Frame disagreed about the upper bound before (closed vs half-open).
    [[nodiscard]] bool is_in_image(const float x, const float y) const
    {
        return x >= min_x && x < max_x && y >= min_y && y < max_y;
    }

    [[nodiscard]] bool has_distortion() const;
    [[nodiscard]] std::vector<cv::Point2f> undistort(const std::vector<cv::Point2f>& points) const;

    cv::Mat K;
    cv::Mat distortion_coefficients;
    int width{};
    int height{};

    float fx{}, fy{}, cx{}, cy{}, invfx{}, invfy{};

    float min_x{}, max_x{}, min_y{}, max_y{};

    float grid_element_width_inv{};
    float grid_element_height_inv{};

    // Quadratic depth-sensor noise coefficient k in sigma_depth(z) = k*z^2 (e.g. Khoshelham &
    // Elberink 2012, Nguyen et al. 2012 for Kinect-style RGB-D sensors). Propagated through
    // inv_depth=1/depth, Var(inv_depth) collapses to k^2 — depth-independent, the classical
    // justification for inverse-depth parameterization. A property of the depth SOURCE, which
    // spans structured light, stereo, LiDAR, ToF and learned monocular depth across the
    // benchmark; read from the calibration's depth_noise_coeff, defaulting to the uncalibrated
    // Kinect-era estimate below (issue #3).
    float depth_noise_coeff{};
};

}
