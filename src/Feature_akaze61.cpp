#include "Feature_akaze61.h"

AF_VSLAM::Akaze61Extractor::Akaze61Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    akaze_options_ = AKAZEOptions();
    // AKAZE's own key of settings/akaze61_settings.yaml: the detector response threshold
    akaze_options_.dthreshold = settings->required_float("FeatureExtractor.detectTh");
}

libAKAZE::AKAZE& AF_VSLAM::Akaze61Extractor::evolution_for(const int width, const int height)
{
    if(evolution_ && width == loaded_width_ && height == loaded_height_)
        return *evolution_;

    // Constructing the object allocates the whole pyramid, so it is worth keeping between
    // frames: Create_Nonlinear_Scale_Space rewrites every per-image buffer, and
    // Feature_Detection clears the keypoint vector it is handed, so nothing survives a frame.
    // What does NOT survive a resize is the allocation itself -- the levels are sized from
    // akaze_options_ in the constructor and never re-checked against the image.
    akaze_options_.img_width = width;
    akaze_options_.img_height = height;

    evolution_.reset();   // free the old pyramid before allocating the new one
    evolution_ = std::make_shared<libAKAZE::AKAZE>(akaze_options_);
    loaded_width_ = width;
    loaded_height_ = height;
    return *evolution_;
}

void AF_VSLAM::Akaze61Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // Build this frame's nonlinear scale space first: AKAZE detects on the evolution, not on
    // the image, so Feature_Detection below reads state that only exists once this has run on
    // this same frame. It used to sit behind a setup_image() hook on FeatureExtractor, called
    // from operator() just before this function -- a virtual that existed for this one
    // extractor out of seven, and that left the evolution null to anything reaching
    // detect_and_compute by another path. Here the ordering cannot be got wrong.
    cv::Mat img_32;
    img.grayImg.convertTo(img_32, CV_32F, 1.0/255.0, 0);

    libAKAZE::AKAZE& evolution = evolution_for(img_32.cols, img_32.rows);
    evolution.Create_Nonlinear_Scale_Space(img_32);

    // AKAZE feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    evolution.Feature_Detection(keypoints_);
    evolution.Compute_Descriptors(keypoints_, descriptors_);

    // Normalize octave and size
    for (int i = 0; i < (int)keypoints_.size(); ++i){
        keypoints_[i].size = 1.0;
        keypoints_[i].octave = 0;
    }

    // Discard keypoints outside the mask (if one was loaded for this frame)
    filter_keypoints_by_mask(keypoints_, descriptors_, img.mask);

    // Retain only if not too many keypoints
    if (keypoints_.size() < settings->oversize_keypoint_factor * settings->max_num_features){
        keypoints = keypoints_;
        descriptors = descriptors_;
        return;
    }

    // Retain best keypoints using octree distribution
    keypoints =  distribute_oct_tree(keypoints_, 0, img.grayImg.cols-1, 0, img.grayImg.rows-1, settings->max_num_features);

    descriptors.create((int)keypoints.size(), descriptors_.cols, descriptors_.type());
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));
}

float AF_VSLAM::Akaze61Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
