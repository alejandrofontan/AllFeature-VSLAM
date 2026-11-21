/**
 * Module: AllFeature-VSLAM - ErrorModel.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - Updated: 2026-11-21
 * - License: GPLv3 License
 *
 * The outlier thresholds every stage tests a residual against: one value for a 2-DoF reprojection error and one
 * for a 3-DoF RGB-D error (pixel + inverse depth), both in squared sigmas. Read once from the settings
 * (Error.Chi2Mono, Error.Chi2RGBD) and used by the optimizer, triangulation, fusion, PnP and the initializer.
 */

#pragma once

#include <cmath>

#include <opencv2/core/persistence.hpp>

#include "SettingsReader.h"
#include "Types.h"

namespace AF_VSLAM
{

struct ErrorModelParameters
{
    float chi2_mono = ChiSquared::dof2_95;   // 2-DoF reprojection error, 95%
    float chi2_rgbd = ChiSquared::dof3_95;   // 3-DoF RGB-D error, 95%

    // Huber deltas of the optimizer: the square roots of the thresholds
    [[nodiscard]] float huber_mono() const { return std::sqrt(chi2_mono); }
    [[nodiscard]] float huber_rgbd() const { return std::sqrt(chi2_rgbd); }
};

class ErrorModel
{
public:
    inline static ErrorModelParameters params{};

    // Call before any other LoadParameters and before the threads start
    static void LoadParameters(const cv::FileStorage& fSettings)
    {
        read_setting_if_present(fSettings, "Error.Chi2Mono", params.chi2_mono);
        read_setting_if_present(fSettings, "Error.Chi2RGBD", params.chi2_rgbd);
    }
};

} // namespace AF_VSLAM
