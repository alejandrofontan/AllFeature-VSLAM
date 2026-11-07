#include "Frame.h"
#include "MapPoint.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AF_VSLAM
{

// ============================================================================
// Construction
// ============================================================================

Frame::Frame(const Image& img, double timestamp_,
    const std::map<FeatureType, std::shared_ptr<FeatureExtractor>>& extractors,
    std::shared_ptr<PlaceRecognition> place_recognition_,
    std::shared_ptr<const Calibration> calibration_):
    timestamp(timestamp_), calibration(std::move(calibration_)),
    frame_id(next_id++), place_recognition(std::move(place_recognition_))
{
    if(place_recognition && place_recognition->needs_image())
        image = img.img;

    extract_features(img, extractors);

    undistort_keypoints();
    get_depth(img);
    get_colors(img);

    for(const auto& [ft, N_] : N){
        pts[ft].assign(N_, nullptr);
        outliers[ft].assign(N_, false);
    }
}

void Frame::extract_features(const Image& img,
                             const std::map<FeatureType, std::shared_ptr<FeatureExtractor>>& extractors)
{
    std::vector<std::pair<FeatureType, FeatureExtractor*>> jobs;
    jobs.reserve(extractors.size());
    for(const auto& [ft, extractor] : extractors)
        jobs.emplace_back(ft, extractor.get());

    struct Out {
        std::vector<cv::KeyPoint> keys;
        cv::Mat desc;
        std::vector<mat2f> sigma2;
        std::vector<mat2f> inf;
        std::vector<float> sizes;
    };
    std::vector<Out> outs(jobs.size());

    std::vector<std::exception_ptr> errors(jobs.size());

    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < static_cast<int>(jobs.size()); ++i) {
        try {
            Out& o = outs[i];
            (*jobs[i].second)(img, o.keys, o.desc, o.sigma2, o.inf, o.sizes);
        } catch(...) {
            errors[i] = std::current_exception();
        }
    }

    for(const std::exception_ptr& error : errors)
        if(error)
            std::rethrow_exception(error);

    feature_types.reserve(jobs.size());
    for(size_t i = 0; i < jobs.size(); ++i) {
        const FeatureType ft = jobs[i].first;
        Out& o = outs[i];

        N[ft] = static_cast<int>(o.keys.size());
        raw_keypoints[ft] = std::move(o.keys);
        descriptors[ft]   = std::move(o.desc);
        keyPtsSigma2[ft]  = std::move(o.sigma2);
        keyPtsInf[ft]     = std::move(o.inf);
        keyPtsSize[ft]    = std::move(o.sizes);

        feature_types.push_back(ft);
    }

    if(!jobs.empty())
        maxKeyPtSize = jobs.front().second->GetMaxKeyPtSize();
}

void Frame::undistort_keypoints()
{
    const bool has_distortion = calibration->has_distortion();

    for(const auto& [ft, N_] : N)
    {
        const std::vector<cv::KeyPoint>& raw = raw_keypoints.at(ft);
        std::vector<cv::KeyPoint>& undistorted = keypoints[ft];
        undistorted = raw;

        if(!has_distortion || N_ == 0)
            continue;

        std::vector<cv::Point2f> pts_in(N_);
        for(int i = 0; i < N_; i++)
            pts_in[i] = raw[i].pt;

        const std::vector<cv::Point2f> pts_out = calibration->undistort(pts_in);
        for(int i = 0; i < N_; i++)
            undistorted[i].pt = pts_out[i];
    }
}

void Frame::get_depth(const Image& img)
{
    const cv::Mat& depth_img = img.depthImg;

    for(const auto& [ft, N_] : N)
    {
        inv_depth[ft].assign(N_, 0.0f);
        sigma2_inv_depth[ft].assign(N_, 0.0f);
    }

    if(depth_img.empty())
        return;

    if(depth_img.type() != CV_32F)
        throw std::runtime_error("Frame::get_depth: unsupported depthImg type: " + std::to_string(depth_img.type()));

    const float k = calibration->depth_noise_coeff;
    const float inv_depth_variance = k * k;

    for(const auto& [ft, N_] : N)
    {
        std::vector<float>& inv_depth_ft = inv_depth.at(ft);
        std::vector<float>& sigma2_inv_depth_ft = sigma2_inv_depth.at(ft);
        const std::vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);

        for(int i = 0; i < N_; i++)
        {
            const int u = cvRound(kps[i].pt.x);
            const int v = cvRound(kps[i].pt.y);

            if(u < 0 || v < 0 || u >= depth_img.cols || v >= depth_img.rows)
                continue;

            const float depth = depth_img.at<float>(v, u);
            if(depth > 0.0f)
            {
                inv_depth_ft[i] = 1.0f / depth;
                sigma2_inv_depth_ft[i] = inv_depth_variance;
            }
        }
    }
}

void Frame::get_colors(const Image& img)
{
    const cv::Mat& im = img.img.empty() ? img.grayImg : img.img;

    for(const auto& [ft, N_] : N)
        keypoint_colors[ft].assign(N_, cv::Vec3b(0, 0, 0));

    if(im.empty())
        return;

    const int channels = im.channels();
    if(im.depth() != CV_8U)
        throw std::runtime_error("Frame::get_colors: unsupported image depth: " + std::to_string(im.depth()));
    if(channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("Frame::get_colors: unsupported number of channels: " + std::to_string(channels));

    for(const auto& [ft, N_] : N)
    {
        std::vector<cv::Vec3b>& colors = keypoint_colors.at(ft);
        const std::vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);

        for(int i = 0; i < N_; i++)
        {
            const int u = cvRound(kps[i].pt.x);
            const int v = cvRound(kps[i].pt.y);
            if(u < 0 || v < 0 || u >= im.cols || v >= im.rows)
                continue;

            switch(channels)
            {
                case 1: {
                    const uchar g = im.at<uchar>(v, u);
                    colors[i] = cv::Vec3b(g, g, g);
                    break;
                }
                case 3:
                    colors[i] = im.at<cv::Vec3b>(v, u);
                    break;
                case 4: {
                    const cv::Vec4b bgra = im.at<cv::Vec4b>(v, u);
                    colors[i] = cv::Vec3b(bgra[0], bgra[1], bgra[2]);
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Pose
// ============================================================================

void Frame::set_pose(const mat4f& Tcw_)
{
    Tcw = Tcw_;
    Rcw = Tcw.block<3,3>(0,0);
    tcw = Tcw.block<3,1>(0,3);
    Rwc = Rcw.transpose();
    twc = -Rwc * tcw;
}

// ============================================================================
// Per-frame queries
// ============================================================================

std::optional<cv::Point2f> Frame::project_if_visible(const Pt& pt, float viewing_cos_limit) const
{
    const Calibration& cal = *calibration;

    const vec3f P = pt->get_world_pos();
    const vec3f Pc = Rcw * P + tcw;
    const float PcX = Pc(0);
    const float PcY = Pc(1);
    const float PcZ = Pc(2);

    if(PcZ <= 0.0f)
        return std::nullopt;

    const float inv_z = 1.0f / PcZ;
    const float u = cal.fx * PcX * inv_z + cal.cx;
    const float v = cal.fy * PcY * inv_z + cal.cy;

    if(u < cal.min_x || u > cal.max_x)
        return std::nullopt;
    if(v < cal.min_y || v > cal.max_y)
        return std::nullopt;

    const float max_distance = pt->get_max_distance_invariance();
    const float min_distance = pt->get_min_distance_invariance();
    const vec3f PO = P - twc;
    const float dist = PO.norm();

    if(dist < min_distance || dist > max_distance)
        return std::nullopt;

    const vec3f Pn = pt->get_normal();
    const float view_cos = PO.dot(Pn) / dist;

    if(view_cos < viewing_cos_limit)
        return std::nullopt;

    return cv::Point2f(u, v);
}

int Frame::count_inlier_map_points() const
{
    int count = 0;
    for (const auto& [ft, num_keypoints] : N) {
        const auto& pts_ft = pts.at(ft);
        const auto& out_ft = outliers.at(ft);
        for(int i = 0; i < num_keypoints; i++)
            if(pts_ft[i] && !out_ft[i])
                ++count;
    }
    return count;
}

void Frame::drop_outlier_points()
{
    for (const auto& [ft, num_keypoints] : N) {
        auto& pts_ft = pts.at(ft);
        auto& out_ft = outliers.at(ft);
        for(int i = 0; i < num_keypoints; i++)
            if(pts_ft[i] && out_ft[i])
            {
                pts_ft[i] = nullptr;
                out_ft[i] = false;
            }
    }
}

void Frame::compute_global_descriptor()
{
    if(place_recognition)
        place_recognition->compute(*this);
}

// ============================================================================
// Spatial index
// ============================================================================

const Frame::Grid& Frame::grid_for(FeatureType feature_type) const
{
    const auto it = grid.find(feature_type);
    if(it != grid.end())
        return it->second;

    Grid& grid_ft = grid[feature_type];
    const std::vector<cv::KeyPoint>& keypoints_ft = keypoints.at(feature_type);
    const int N_ = N.at(feature_type);

    for(int i = 0; i < N_; i++)
    {
        const cv::Point2i cell = pos_in_grid(keypoints_ft[i]);
        grid_ft[cell.x][cell.y].push_back(static_cast<std::size_t>(i));
    }

    return grid_ft;
}

void Frame::build_grids() const
{
    for(const auto& [ft, N_] : N)
        grid_for(ft);
}

cv::Point2i Frame::pos_in_grid(const cv::KeyPoint& kp) const
{
    const Calibration& cal = *calibration;

    const int x = static_cast<int>(std::floor((kp.pt.x - cal.min_x) * cal.grid_element_width_inv));
    const int y = static_cast<int>(std::floor((kp.pt.y - cal.min_y) * cal.grid_element_height_inv));

    return {std::clamp(x, 0, FRAME_GRID_COLS - 1), std::clamp(y, 0, FRAME_GRID_ROWS - 1)};
}

std::vector<std::size_t> Frame::get_features_in_area(const float x, const float y, const float r,
                                                     const FeatureType feature_type) const
{
    return features_in_area(grid_for(feature_type), keypoints.at(feature_type), *calibration, x, y, r);
}

std::vector<std::size_t> features_in_area(const Frame::Grid& grid,
                                          const std::vector<cv::KeyPoint>& keypoints,
                                          const Calibration& cal,
                                          const float x, const float y, const float r)
{
    std::vector<std::size_t> indices;
    indices.reserve(keypoints.size());

    const int min_cell_x = std::max(0, static_cast<int>(std::floor((x - cal.min_x - r) * cal.grid_element_width_inv)));
    if(min_cell_x >= FRAME_GRID_COLS)
        return indices;

    const int max_cell_x = std::min(FRAME_GRID_COLS - 1, static_cast<int>(std::floor((x - cal.min_x + r) * cal.grid_element_width_inv)));
    if(max_cell_x < 0)
        return indices;

    const int min_cell_y = std::max(0, static_cast<int>(std::floor((y - cal.min_y - r) * cal.grid_element_height_inv)));
    if(min_cell_y >= FRAME_GRID_ROWS)
        return indices;

    const int max_cell_y = std::min(FRAME_GRID_ROWS - 1, static_cast<int>(std::floor((y - cal.min_y + r) * cal.grid_element_height_inv)));
    if(max_cell_y < 0)
        return indices;

    const float r2 = r * r;

    for(int cell_x = min_cell_x; cell_x <= max_cell_x; cell_x++)
    {
        for(int cell_y = min_cell_y; cell_y <= max_cell_y; cell_y++)
        {
            for(const std::size_t index : grid[cell_x][cell_y])
            {
                const cv::KeyPoint& kp = keypoints[index];
                const float dist_x = kp.pt.x - x;
                const float dist_y = kp.pt.y - y;

                if(dist_x * dist_x + dist_y * dist_y < r2)
                    indices.push_back(index);
            }
        }
    }

    return indices;
}

}
