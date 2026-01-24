#include "Feature_aliked128.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

ANYFEATURE_VSLAM::FeatureExtractor_aliked128::FeatureExtractor_aliked128(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
            
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;    
        extractor = std::make_shared<ALIKED>("aliked-n16", device.str());
}

static cv::Mat tensorDescToMatCopy(const at::Tensor& desc_in) {
    // Move to CPU, contiguous, float32
    at::Tensor desc = desc_in;
    if (desc.is_cuda()) desc = desc.to(at::kCPU);
    desc = desc.contiguous();
    if (desc.scalar_type() != at::kFloat) desc = desc.to(at::kFloat);

    // Handle [1,N,D] -> [N,D]
    if (desc.dim() == 3 && desc.size(0) == 1) {
        desc = desc.squeeze(0);
    }

    TORCH_CHECK(desc.dim() == 2, "Expected descriptors [N,D] or [1,N,D]");
    const int N = (int)desc.size(0);
    const int D = (int)desc.size(1);

    // Create independent cv::Mat and copy
    cv::Mat out(N, D, CV_32F);
    std::memcpy(out.data, desc.data_ptr<float>(), (size_t)N * (size_t)D * sizeof(float));
    return out;
}

void ANYFEATURE_VSLAM::FeatureExtractor_aliked128::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    cv::Mat img_;
    if (img.img.channels() == 3) {
        img_ = img.img.clone();
    } else if (img.img.channels() == 1) {
        cv::cvtColor(img.img, img_, cv::COLOR_GRAY2RGB); 
    } else if (img.img.channels() == 4) {
        cv::cvtColor(img.img, img_, cv::COLOR_BGRA2RGB);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }

    auto feats0 = extractor->run(img_);

    const auto& kpts = feats0.at("keypoints");
    at::Tensor kpts_cpu = kpts.cpu().contiguous().to(at::kFloat);
    TORCH_CHECK(kpts_cpu.dim() == 2 && kpts_cpu.size(1) >= 2, "Expected keypoints [N,2] (or more)");

    const auto& scores_t = feats0.at("scores"); 
    at::Tensor scores_cpu = scores_t.squeeze().cpu().contiguous().to(at::kFloat);
    TORCH_CHECK(scores_cpu.dim() == 1, "Expected scores [N] after squeeze()");
    TORCH_CHECK(scores_cpu.size(0) == kpts_cpu.size(0), "scores N != keypoints N");
    auto sacc = scores_cpu.accessor<float, 1>();

    int N = (int)kpts_cpu.size(0);
    auto acc = kpts_cpu.accessor<float, 2>();
    int iKey{0};
    for (int i = 0; i < N; ++i) {
        float x = acc[i][0];
        float y = acc[i][1];
        cv::KeyPoint keyPt{};
        keyPt.pt.x = x;
        keyPt.pt.y = y;
        keyPt.class_id = iKey;
        keyPt.size = 1;
        keyPt.angle = 0;
        keyPt.octave = 0;
        keyPt.response = sacc[i];
        keypoints.push_back(keyPt);
        ++iKey;
    }

    const auto& desc_t = feats0.at("descriptors");
    descriptors = tensorDescToMatCopy(desc_t);

    // float min_resp = 0.2f;  // choose your threshold
    // std::vector<cv::KeyPoint> kps_f;
    // kps_f.reserve(keypoints.size());

    // // We assume descriptors is CV_32F and has N rows.
    // // If it's binary / CV_8U, this still works (row copy is type-agnostic).
    // cv::Mat desc_f;
    // desc_f.create(0, descriptors.cols, descriptors.type());

    // for (int i = 0; i < (int)keypoints.size(); ++i) {
    //     if (keypoints[i].response >= min_resp) {
    //         kps_f.push_back(keypoints[i]);
    //         desc_f.push_back(descriptors.row(i));  // keep matching row
    //     }
    // }

    // // Replace originals
    // keypoints.swap(kps_f);
    // descriptors = desc_f;
}

int ANYFEATURE_VSLAM::FeatureExtractor_aliked128::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return keypoint.octave;
}

float ANYFEATURE_VSLAM::FeatureExtractor_aliked128::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return powf(settings->GetDetectorScaleFactor(), float(GetKeypointOctave(keypoint)));
}

float ANYFEATURE_VSLAM::DescriptorDistance_aliked128(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2);
}