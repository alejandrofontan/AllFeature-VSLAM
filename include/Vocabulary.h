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
        Vocabulary(const string vocabularyFolder, const  FeatureType featureType);

        FeatureType featureType;
        const string vocabularyFolder;

        // Vocabulary variables
        std::shared_ptr<AnyFeatNonBinVocabulary> anyFeatNonBinVocabulary;
        std::shared_ptr<AnyFeatBinVocabulary> anyFeatBinVocabulary;
        std::shared_ptr<R2d2Vocabulary> r2d2Vocabulary;
        std::shared_ptr<Sift128Vocabulary> sift128Vocabulary;
        std::shared_ptr<Kaze64Vocabulary> kaze64Vocabulary;
        std::shared_ptr<Surf64Vocabulary> surf64Vocabulary;
        std::shared_ptr<BriskVocabulary> briskVocabulary;
        std::shared_ptr<Akaze61Vocabulary> akaze61Vocabulary;
        std::shared_ptr<OrbVocabulary> orbVocabulary;

        void createVocabulary();
        bool loadFromTextFile();
        unsigned int size();
        void transform(const cv::Mat& mDescriptors, DBoW2::BowVector& mBowVec,DBoW2::FeatureVector& mFeatVec);
        double score(const DBoW2::BowVector &BowVec_1, const DBoW2::BowVector &BowVec_2 );

    };
}

#endif //AF_VSLAM_VOCABULARY_H
