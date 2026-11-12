#ifndef AF_VSLAM_FEATUREEXTRACTOR_H
#define AF_VSLAM_FEATUREEXTRACTOR_H

#include "Types.h"
#include "Definitions.h"
#include "Image.h"
#include "KeypointData.h"

#include <list>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>

#include <opencv2/core.hpp>

namespace AF_VSLAM {

    class FeatureExtractor;
    class ExtractorNode;

    // One feature type's per-keypoint scale data, as an extractor produces it.
    //
    class ExtractorNode;

    // One feature type's per-keypoint scale data, as an extractor produces it.
    //
    // Today all three come from the same scalar: computeSize fills `size` with s, then
    // computeSigma fills sigma2 with s^2*I and information with (1/s^2)*I. They are stored
    // independently, not derived, so that a size and a sigma can differ and so that sigma can
    // be genuinely 2D -- a decision to keep, even though the only producer so far is isotropic.
    struct KeypointScalesPerType
    {
        std::vector<mat2f> sigma2;
        std::vector<mat2f> information;
        std::vector<float> size;
    };

    // Every feature type's scale data, as Frame and KeyFrame hold it. Defined here, beside the
    // code that fills it, so the producer and the two consumers cannot disagree: Frame and
    // KeyFrame used to each declare the three maps AND their own copy of these accessors.
    //
    // `size` is no longer read directly -- it exists to build sigma2 and information. It stopped
    // being a distance quantity when MapPoint's depth window became a plain ratio.
    struct KeypointScales
    {
        std::map<FeatureType, KeypointScalesPerType> per_type;

        // trace/2 -- the isotropic summary of what may be an anisotropic covariance, not an
        // average of a number with itself (which is all it looks like while the one producer
        // fills both diagonal entries equally).
        [[nodiscard]] float get_sigma2(KeypointIndex keypoint_index, FeatureType feature_type) const
        {
            const mat2f& m = element(at(feature_type).sigma2, keypoint_index, feature_type);
            return 0.5f * (m(0,0) + m(1,1));
        }

        [[nodiscard]] mat2f get_information_2d(KeypointIndex keypoint_index, FeatureType feature_type) const
        {
            return element(at(feature_type).information, keypoint_index, feature_type);
        }

        // std::map::at and std::vector::at throw a std::out_of_range reading only "map::at" or
        // "vector::at" -- in a crash log that names neither which of the two lookups failed, nor
        // the feature type or the index. Nothing catches it, so that bare message is all the run
        // leaves behind. Same reason KeyFrame.cc wraps its map-point slots.
        [[nodiscard]] const KeypointScalesPerType& at(FeatureType feature_type) const
        {
            const auto it = per_type.find(feature_type);
            if(it == per_type.end())
                throw std::out_of_range("KeypointScales: no feature type "
                                        + std::to_string(static_cast<int>(feature_type)));
            return it->second;
        }

        [[nodiscard]] static const mat2f& element(const std::vector<mat2f>& values,
                                                  KeypointIndex keypoint_index, FeatureType feature_type)
        {
            if(keypoint_index < 0 || static_cast<std::size_t>(keypoint_index) >= values.size())
                throw std::out_of_range("KeypointScales: feature type "
                                        + std::to_string(static_cast<int>(feature_type))
                                        + ", keypoint index " + std::to_string(keypoint_index)
                                        + " outside [0, " + std::to_string(values.size()) + ")");
            return values[static_cast<std::size_t>(keypoint_index)];
        }
    };

    class ExtractorNode
    {
    public:
        ExtractorNode():bNoMore(false){}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    // What every extractor reads from its feature's settings yaml: the keypoint budget. Anything else is specific to
    // one extractor, which reads it itself through required_int / required_float - so a key in a feature's yaml is a
    // key that feature uses, and a parameter added to one extractor touches no other yaml. (nOctaves, scaleFactor
    // and detectTh used to be read here for all eight features; only ORB uses the first two, ORB and AKAZE the third.)
    class FeatureExtractorSettings
    {
    public:

        int maxNumFeatures;

        FeatureExtractorSettings(const FeatureType& featureType__,
                                 const std::string &settingsYamlFile);

        // A key of the settings yaml this extractor cannot run without.
        // @throws std::runtime_error naming the file and the key when it is missing or not a number.
        [[nodiscard]] int required_int(const std::string& key) const;
        [[nodiscard]] float required_float(const std::string& key) const;

        std::string settingsYamlFile;

        VerbosityLevel verbosity{LOW};

        FeatureType featureType;

        const float OVERSIZE_KEYPOINT_FACTOR{1.05f};

    };

    class FeatureExtractor {

    public:

        FeatureExtractor(std::shared_ptr<FeatureExtractorSettings>& settings_): settings(settings_){};
        virtual ~FeatureExtractor() = default;

        std::shared_ptr<FeatureExtractorSettings> settings{};
        std::vector<double> extractorTime{};

        void operator()(const Image& img, KeypointData& data);

    protected:
        void computeSize(KeypointData& data);
        void computeSigma(KeypointData& data);

        virtual void setupImage(const Image&){};

    protected:
        [[nodiscard]] virtual int GetKeypointOctave(const cv::KeyPoint& keypoint) const = 0;
        [[nodiscard]] virtual float GetKeypointSize(const cv::KeyPoint& keypoint) const = 0;

        virtual void detectAndCompute(const Image&, std::vector<cv::KeyPoint>&, cv::Mat&){};
        std::vector<cv::KeyPoint> DistributeOctTree(std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
            const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);
        void FilterKeypointsByMask(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask) const;
    };
}
#endif //AF_VSLAM_FEATUREEXTRACTOR_H
