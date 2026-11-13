#include "Feature_akaze61.h"

AF_VSLAM::Akaze61Extractor::Akaze61Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    akaze_options = AKAZEOptions();
    // AKAZE's own key of settings/akaze61_settings.yaml: the detector response threshold
    akaze_options.dthreshold = settings->required_float("FeatureExtractor.detectTh");
}

void AF_VSLAM::Akaze61Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // AKAZE feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    evolution->Feature_Detection(keypoints_);
    evolution->Compute_Descriptors(keypoints_, descriptors_);

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

void AF_VSLAM::Akaze61Extractor::setup_image(const Image& img){
    cv::Mat img_32;
    img.grayImg.convertTo(img_32, CV_32F, 1.0/255.0,0);
    akaze_options.img_width = img_32.cols;
    akaze_options.img_height = img_32.rows;

    evolution.reset();
    evolution = std::make_shared<libAKAZE::AKAZE>(akaze_options);
    evolution->Create_Nonlinear_Scale_Space(img_32);
}

float AF_VSLAM::Akaze61Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
