//
// Created by fontan on 18/03/24.
//

#include "Vocabulary.h"
#include "Converter.h"
#include "FeatureFactory.h"

ANYFEATURE_VSLAM::Vocabulary::Vocabulary(const string vocabularyFolder, const FeatureType featureType):
        vocabularyFolder(vocabularyFolder), featureType(featureType)
{}

void ANYFEATURE_VSLAM::Vocabulary::createVocabulary(){
    switch(featureType) {
        // Create Vocabulary
        case FEAT_ANYFEATNONBIN: {
            anyFeatNonBinVocabulary = make_shared<AnyFeatNonBinVocabulary>();
            return;
        }
        case FEAT_ANYFEATBIN: {
            anyFeatBinVocabulary = make_shared<AnyFeatBinVocabulary>();
            return;
        }
        case FEAT_R2D2: {
            r2d2Vocabulary = make_shared<R2d2Vocabulary>();
            return;
        }
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
        case FEAT_BRISK: {
            briskVocabulary = make_shared<BriskVocabulary>();
            return;
        }
        case FEAT_AKAZE61: {
            akaze61Vocabulary = make_shared<Akaze61Vocabulary>();
            return;
        }
        case FEAT_ORB: {
            orbVocabulary = make_shared<OrbVocabulary>();
            return;
        }
    }
}

bool ANYFEATURE_VSLAM::Vocabulary::loadFromTextFile(){
    switch(featureType) {
        // load from text file
        case FEAT_ANYFEATNONBIN:{
            cout << "Loading AnyFeatNonBin Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/AnyFeatNonBin_DBoW2_voc.txt";
            return anyFeatNonBinVocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_ANYFEATBIN:{
            cout << "Loading AnyFeatBin Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/AnyFeatBin_DBoW2_voc.txt";
            return anyFeatBinVocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_R2D2:{
            cout << "Loading R2d2 Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/R2d2_DBoW2_voc.txt";
            return r2d2Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_SIFT128:{
            cout << "Loading Sift128 Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/Sift128_DBoW2_voc.txt";
            return sift128Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_KAZE64:{
            cout << "Loading Kaze64 Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/Kaze64_DBoW2_voc.txt";
            return kaze64Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_SURF64:{
            cout << "Loading Surf64 Vocabulary. This could take a while..." << endl;
            string vocabulary_path = vocabularyFolder + "/Surf64_DBoW2_voc.txt";
            return surf64Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_BRISK:{
            string vocabulary_path = vocabularyFolder + "/Brisk_DBoW2_voc.txt";
            cout << "Loading Brisk Vocabulary from: " + vocabulary_path << endl;
            cout << "This could take a while ..." << endl;
            return briskVocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_AKAZE61:{
            string vocabulary_path = vocabularyFolder + "/Akaze61_DBoW2_voc.txt";
            cout << "Loading Akaze61 Vocabulary from: " + vocabulary_path << endl;
            cout << "This could take a while ..." << endl;
            return akaze61Vocabulary->loadFromTextFile(vocabulary_path);
        }
        case FEAT_ORB:{
            string vocabulary_path = vocabularyFolder + "/ORBvoc.txt";
            cout << "Loading Orb Vocabulary from: " + vocabulary_path << endl;
            cout << "This could take a while ..." << endl;
            return orbVocabulary->loadFromTextFile(vocabulary_path);
        }
    }
}

unsigned int ANYFEATURE_VSLAM::Vocabulary::size(){
    switch(featureType) {
        // size
        case FEAT_ANYFEATNONBIN:
            return anyFeatNonBinVocabulary->size();
        case FEAT_ANYFEATBIN:
            return anyFeatBinVocabulary->size();
        case FEAT_R2D2:
            return r2d2Vocabulary->size();
        case FEAT_SIFT128:
            return sift128Vocabulary->size();
        case FEAT_KAZE64:
            return kaze64Vocabulary->size();
        case FEAT_SURF64:
            return surf64Vocabulary->size();
        case FEAT_BRISK:
            return briskVocabulary->size();
        case FEAT_AKAZE61:
            return akaze61Vocabulary->size();
        case FEAT_ORB:
            return orbVocabulary->size();
    }
}

double ANYFEATURE_VSLAM::Vocabulary::score(const DBoW2::BowVector &BowVec_1, const DBoW2::BowVector &BowVec_2){
    switch(featureType) {
        // score
        case FEAT_ANYFEATNONBIN:
            return anyFeatNonBinVocabulary->score(BowVec_1,BowVec_2);
        case FEAT_ANYFEATBIN:
            return anyFeatBinVocabulary->score(BowVec_1,BowVec_2);
        case FEAT_R2D2:
            return r2d2Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_SIFT128:
            return sift128Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_KAZE64:
            return kaze64Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_SURF64:
            return surf64Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_BRISK:
            return briskVocabulary->score(BowVec_1,BowVec_2);
        case FEAT_AKAZE61:
            return akaze61Vocabulary->score(BowVec_1,BowVec_2);
        case FEAT_ORB:
            return orbVocabulary->score(BowVec_1, BowVec_2);
    }
}

void ANYFEATURE_VSLAM::Vocabulary::transform(
        const cv::Mat& mDescriptors,  DBoW2::BowVector& mBowVec,DBoW2::FeatureVector& mFeatVec){
    switch(featureType) {
        // transform
        case FEAT_ANYFEATNONBIN:{
            vector<vector<float>> vCurrentDesc = Converter::toDescriptorVector_float(mDescriptors);
            anyFeatNonBinVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_ANYFEATBIN:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            anyFeatBinVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_R2D2:{
            vector<vector<float>> vCurrentDesc = Converter::toDescriptorVector_float(mDescriptors);
            r2d2Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
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
        case FEAT_BRISK:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            briskVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_AKAZE61:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            akaze61Vocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
        case FEAT_ORB:{
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector_mat(mDescriptors);
            orbVocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
            return;
        }
    }
}

