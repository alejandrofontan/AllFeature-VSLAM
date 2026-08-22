//
// Created by fontan on 18/03/24.
//

#ifndef AF_VSLAM_VOCABULARY_H
#define AF_VSLAM_VOCABULARY_H

#include "FeatureFactory.h"
#include "Definitions.h"
#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/DBoW2.h"

namespace AF_VSLAM {
    class Vocabulary {
    public:
        Vocabulary(const string vocabularyFolder, const FeatureType featureType, const bool enabled = true);

        FeatureType featureType;
        const string vocabularyFolder;

        // Backend on/off switch, decided at startup by System's VPR resolution
        // (settings key `vpr:`). Distinct from has_vocabulary(): a classical feature
        // can be deliberately disabled with `vpr: none`.
        const bool enabled;

        // Vocabulary variables
        std::shared_ptr<Sift128Vocabulary> sift128Vocabulary;
        std::shared_ptr<Kaze64Vocabulary> kaze64Vocabulary;
        std::shared_ptr<Surf64Vocabulary> surf64Vocabulary;
        std::shared_ptr<BriskVocabulary> briskVocabulary;
        std::shared_ptr<Akaze61Vocabulary> akaze61Vocabulary;
        std::shared_ptr<OrbVocabulary> orbVocabulary;

        // Whether a DBoW2 vocabulary exists for this feature type (false for the
        // learned features aliked128/superpoint256).
        static bool has_vocabulary(FeatureType featureType);
        bool isSupported() const;

        // The single gate every BoW consumer checks: enabled AND a vocabulary exists.
        // When false the system runs without VPR — no loop closing, no relocalization
        // (a tracking loss is then permanent).
        bool is_active() const;

        void createVocabulary();
        bool loadFromTextFile();
        unsigned int size();
        void transform(const cv::Mat& mDescriptors, DBoW2::BowVector& mBowVec,DBoW2::FeatureVector& mFeatVec);
        double score(const DBoW2::BowVector &BowVec_1, const DBoW2::BowVector &BowVec_2 );

    };
}

#endif //AF_VSLAM_VOCABULARY_H
