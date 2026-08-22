//
// Created by fontan on 18/03/24.
//

#include "Vocabulary.h"
#include "Converter.h"
#include "FeatureFactory.h"
#include "afvslam_log.hpp"

#include <atomic>

AF_VSLAM::Vocabulary::Vocabulary(const string vocabularyFolder, const FeatureType featureType, const bool enabled):
        featureType(featureType), vocabularyFolder(vocabularyFolder), enabled(enabled)
{}

void AF_VSLAM::Vocabulary::createVocabulary(){
    switch(featureType) {
        // Create Vocabulary
        case FEAT_SIFT128: {
            sift128Vocabulary = make_shared<Sift128Vocabulary>();
            return;
        }
        case FEAT_KAZE64: {
            kaze64Vocabulary = make_shared<Kaze64Vocabulary>();
            return;
        }
        case FEAT_SURF64: {
            surf64Vocabulary = make_shared<Surf64Vocabulary>();
            return;
        }
        case FEAT_BRISK48: {
            briskVocabulary = make_shared<BriskVocabulary>();
            return;
        }
        case FEAT_AKAZE61: {
            akaze61Vocabulary = make_shared<Akaze61Vocabulary>();
            return;
        }
        case FEAT_ORB32: {
            orbVocabulary = make_shared<OrbVocabulary>();
            return;
        }
        case FEAT_ALIKED128:
        case FEAT_SUPERPOINT256:
            AF_ERROR("[Vocabulary] createVocabulary: no DBoW2 vocabulary support yet for feature type " + get_feature(featureType).getFeatureName());
            return;
    }
}

bool AF_VSLAM::Vocabulary::loadFromTextFile(){
    switch(featureType) {
        // load from text file
        case FEAT_SIFT128:{
            string vocabulary_path = vocabularyFolder + "/Sift128_DBoW2_voc.txt";
            AF_INFO("Loading Sift128 Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return sift128Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_KAZE64:{
            string vocabulary_path = vocabularyFolder + "/Kaze64_DBoW2_voc.txt";
            AF_INFO("Loading Kaze64 Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return kaze64Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_SURF64:{
            string vocabulary_path = vocabularyFolder + "/Surf64_DBoW2_voc.txt";
            AF_INFO("Loading Surf64 Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return surf64Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_BRISK48:{
            string vocabulary_path = vocabularyFolder + "/Brisk_DBoW2_voc.txt";
            AF_INFO("Loading Brisk Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return briskVocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_AKAZE61:{
            string vocabulary_path = vocabularyFolder + "/Akaze61_DBoW2_voc.txt";
            AF_INFO("Loading Akaze61 Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return akaze61Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_ORB32:{
            string vocabulary_path = vocabularyFolder + "/ORBvoc.txt";
            AF_INFO("Loading Orb Vocabulary from: " + vocabulary_path);
            AF_INFO("This could take a while ...");
            return orbVocabulary->loadFromTextFile(vocabulary_path);
        }
        default:
            AF_ERROR("[Vocabulary] loadFromTextFile: no DBoW2 vocabulary support yet for feature type " + get_feature(featureType).getFeatureName());
            return false;
    }
}

unsigned int AF_VSLAM::Vocabulary::size(){
    switch(featureType) {
        // size
        case FEAT_SIFT128:
            return sift128Vocabulary->size();
        case FEAT_KAZE64:
            return kaze64Vocabulary->size();
        case FEAT_SURF64:
            return surf64Vocabulary->size();
        case FEAT_BRISK48:
            return briskVocabulary->size();
        case FEAT_AKAZE61:
            return akaze61Vocabulary->size();
        case FEAT_ORB32:
            return orbVocabulary->size();
        default:
            AF_ERROR("[Vocabulary] size: no DBoW2 vocabulary support yet for feature type " + get_feature(featureType).getFeatureName());
            return 0;
    }
}

double AF_VSLAM::Vocabulary::score(const DBoW2::BowVector &BowVec_1, const DBoW2::BowVector &BowVec_2){
    switch(featureType) {
        // score
        case FEAT_SIFT128:
            return sift128Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_KAZE64:
            return kaze64Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_SURF64:
            return surf64Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_BRISK48:
            return briskVocabulary->score(BowVec_1,BowVec_2);
        case FEAT_AKAZE61:
            return akaze61Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_ORB32:
            return orbVocabulary->score(BowVec_1, BowVec_2);
        default:
            AF_ERROR("[Vocabulary] score: no DBoW2 vocabulary support yet for feature type " + get_feature(featureType).getFeatureName());
            return 0.0;
    }
}

void AF_VSLAM::Vocabulary::transform(
        const cv::Mat& mDescriptors,  DBoW2::BowVector& mBowVec,DBoW2::FeatureVector& mFeatVec){
    switch(featureType) {
        // transform
        case FEAT_SIFT128:{
            vector<vector<float>> vCurrentDesc = Converter::toDescriptorVector_float(mDescriptors);
            sift128Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_KAZE64:{
            vector<vector<float>> vCurrentDesc = Converter::toDescriptorVector_float(mDescriptors);
            kaze64Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_SURF64:{
            vector<vector<float>> vCurrentDesc = Converter::toDescriptorVector_float(mDescriptors);
            surf64Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_BRISK48:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            briskVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_AKAZE61:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            akaze61Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_ORB32:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            orbVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_ALIKED128:
        case FEAT_SUPERPOINT256: {
            // No BoW for learned features — expected when running vocabulary-less (see
            // Vocabulary::isSupported). Warn once instead of erroring per keyframe.
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                AF_WARN("[Vocabulary] transform: no DBoW2 vocabulary for "
                        + get_feature(featureType).getFeatureName()
                        + " — BoW vectors stay empty (loop closing / BoW relocalization inactive)");
            return;
        }
    }
}

bool AF_VSLAM::Vocabulary::has_vocabulary(const FeatureType featureType) {
    switch (featureType) {
        case FEAT_SIFT128:
        case FEAT_KAZE64:
        case FEAT_SURF64:
        case FEAT_BRISK48:
        case FEAT_AKAZE61:
        case FEAT_ORB32:
            return true;
        default:
            return false;
    }
}

bool AF_VSLAM::Vocabulary::isSupported() const {
    return has_vocabulary(featureType);
}

bool AF_VSLAM::Vocabulary::is_active() const {
    return enabled && has_vocabulary(featureType);
}

