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

    for(const auto& [ft, data] : features.per_type){
        pts[ft].assign(data.count(), nullptr);
        outliers[ft].assign(data.count(), false);
    }
}

void Frame::extract_features(const Image& img,
                             const std::map<FeatureType, std::shared_ptr<FeatureExtractor>>& extractors)
{
    std::vector<std::pair<FeatureType, FeatureExtractor*>> jobs;
    jobs.reserve(extractors.size());
    for(const auto& [ft, extractor] : extractors)
        jobs.emplace_back(ft, extractor.get());

    std::vector<KeypointData> outs(jobs.size());

    std::vector<std::exception_ptr> errors(jobs.size());

    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < static_cast<int>(jobs.size()); ++i) {
        try {
            (*jobs[i].second)(img, outs[i]);
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

        raw_keypoints[ft] = outs[i].keypoints;
        features.per_type[ft] = std::move(outs[i]);

        feature_types.push_back(ft);
    }
}

void Frame::undistort_keypoints()
{
    const bool has_distortion = calibration->has_distortion();

    for(auto& [ft, data] : features.per_type)
    {
        const std::vector<cv::KeyPoint>& raw = raw_keypoints.at(ft);
        const int N_ = static_cast<int>(raw.size());
        std::vector<cv::KeyPoint>& undistorted = data.keypoints;

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

    for(auto& [ft, data] : features.per_type)
    {
        data.inv_depth.assign(data.count(), 0.0f);
        data.sigma2_inv_depth.assign(data.count(), 0.0f);
    }

    if(depth_img.empty())
        return;

    if(depth_img.type() != CV_32F)
        throw std::runtime_error("Frame::get_depth: unsupported depthImg type: " + std::to_string(depth_img.type()));

    const float k = calibration->depth_noise_coeff;
    const float inv_depth_variance = k * k;

    for(auto& [ft, data] : features.per_type)
    {
        std::vector<float>& inv_depth_ft = data.inv_depth;
        std::vector<float>& sigma2_inv_depth_ft = data.sigma2_inv_depth;
        const std::vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);
        const int N_ = static_cast<int>(kps.size());

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

    for(auto& [ft, data] : features.per_type)
        data.colors.assign(data.count(), cv::Vec3b(0, 0, 0));

    if(im.empty())
        return;

    const int channels = im.channels();
    if(im.depth() != CV_8U)
        throw std::runtime_error("Frame::get_colors: unsupported image depth: " + std::to_string(im.depth()));
    if(channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("Frame::get_colors: unsupported number of channels: " + std::to_string(channels));

    for(auto& [ft, data] : features.per_type)
    {
        std::vector<cv::Vec3b>& colors = data.colors;
        const std::vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);
        const int N_ = static_cast<int>(kps.size());

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
    pose_.set(Tcw_);
}

// ============================================================================
// Per-frame queries
// ============================================================================

std::optional<Projection> project_if_visible(const Pose& pose, const Calibration& cal,
                                             const Pt& pt, const float viewing_cos_limit)
{
    const vec3f P = pt->get_world_pos();
    const vec3f Pc = pose.Rcw * P + pose.tcw;
    const float PcX = Pc(0);
    const float PcY = Pc(1);
    const float PcZ = Pc(2);

    if(PcZ <= 0.0f)
        return std::nullopt;

    const float inv_z = 1.0f / PcZ;
    const float u = cal.fx * PcX * inv_z + cal.cx;
    const float v = cal.fy * PcY * inv_z + cal.cy;

    if(!cal.is_in_image(u, v))
        return std::nullopt;

    const float max_distance = pt->get_max_distance_invariance();
    const float min_distance = pt->get_min_distance_invariance();
    const vec3f PO = P - pose.twc;
    const float dist = PO.norm();

    if(dist < min_distance || dist > max_distance)
        return std::nullopt;

    const vec3f Pn = pt->get_normal();
    const float view_cos = PO.dot(Pn) / dist;

    if(view_cos < viewing_cos_limit)
        return std::nullopt;

    return Projection{cv::Point2f(u, v), dist};
}

std::optional<cv::Point2f> Frame::project_if_visible(const Pt& pt, float viewing_cos_limit) const
{
    if(const std::optional<Projection> projection =
           AF_VSLAM::project_if_visible(pose_, *calibration, pt, viewing_cos_limit))
        return projection->uv;
    return std::nullopt;
}

int Frame::count_inlier_map_points() const
{
    int count = 0;
    for (const auto& [ft, pts_ft] : pts) {
        const auto& out_ft = outliers.at(ft);
        for(std::size_t i = 0; i < pts_ft.size(); i++)
            if(pts_ft[i] && !out_ft[i])
                ++count;
    }
    return count;
}

void Frame::drop_outlier_points()
{
    for (auto& [ft, pts_ft] : pts) {
        auto& out_ft = outliers.at(ft);
        for(std::size_t i = 0; i < pts_ft.size(); i++)
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
    const std::vector<cv::KeyPoint>& keypoints_ft = features.at(feature_type).keypoints;
    const int N_ = static_cast<int>(keypoints_ft.size());

    for(int i = 0; i < N_; i++)
    {
        const cv::Point2i cell = pos_in_grid(keypoints_ft[i]);
        grid_ft[cell.x][cell.y].push_back(static_cast<std::size_t>(i));
    }

    return grid_ft;
}

void Frame::build_grids() const
{
    for(const auto& [ft, data] : features.per_type)
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
    return features_in_area(grid_for(feature_type), features.at(feature_type).keypoints, *calibration, x, y, r);
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
