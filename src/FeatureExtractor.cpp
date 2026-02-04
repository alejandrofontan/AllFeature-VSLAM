#include "FeatureExtractor.h"
#include "Utils.h"
#include "MathFunctions.h"

#include <opencv2/core/core.hpp>
#include <vector>

#include <yaml-cpp/yaml.h>

using namespace cv;
using namespace std;

ANYFEATURE_VSLAM::FeatureExtractorSettings::FeatureExtractorSettings(
        const KeypointType& keypointType_, const DescriptorType& descriptorType_,
        const std::string &settingsYamlFile):
        keypointType(keypointType_), descriptorType(descriptorType_){

    YAML::Node settings = YAML::LoadFile(settingsYamlFile);
    nOctaves = settings["FeatureExtractor.nOctaves"].as<int>();
    scaleFactor = settings["FeatureExtractor.scaleFactor"].as<float>();
    detectTh = settings["FeatureExtractor.detectTh"].as<float>();
    maxNumFeatures = settings["FeatureExtractor.maxNumFeatures"].as<int>();
    
    cout << endl  << "Loading Feature Extractor Settings from : " << settingsYamlFile << endl;
    std::cout << "- nOctaves = " << nOctaves << std::endl;
    std::cout << "- scaleFactor = " << scaleFactor << std::endl;
    std::cout << "- detectTh = " << detectTh << std::endl; 
    std::cout << "- maxNumFeatures = " << maxNumFeatures << std::endl;       

    maxKeyPtSize0  = pow(scaleFactorOrb,float(nOctavesOrb - 1.0));
    maxKeyPtSigma0 = pow(scaleFactorOrb,float(nOctavesOrb - 1.0));
    maxKeyPtSize = pow(scaleFactorOrb,float(nOctavesOrb - 1.0));
    minKeyPtSize = 1.0f;
}

void ANYFEATURE_VSLAM::FeatureExtractor::operator()(const Image& img,
                                             std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                                             std::vector<mat2f>& keyPtsSigma2, std::vector<mat2f>& keyPtsInf, std::vector<float>& keyPtsSize)
{
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    setupImage(img);
    detectAndCompute(img, keypoints, descriptors);
    computeSize(keyPtsSize,keypoints);
    computeSigma(keyPtsSigma2, keyPtsInf,keyPtsSize,keypoints,img,CovarianceMethod::SIZE);

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
    //medianTrackingTime(t_duration, extractorTime, "        - Extractor Time   ", TRACKING_PROFILING); 
}

void ANYFEATURE_VSLAM::FeatureExtractor::computeSize(std::vector<float>& keyPtsSize, const std::vector<cv::KeyPoint>& keypoints){
    keyPtsSize.clear();
    keyPtsSize.reserve(keypoints.size());
    for(auto& keyPt: keypoints){
        float keyPtSize = GetKeypointSize(keyPt);
        float keyPtSize_norm{settings->maxKeyPtSize};
        if(settings->maxKeyPtSize > settings->minKeyPtSize)
            keyPtSize_norm =  1.0f +  (keyPtSize - settings->minKeyPtSize) * (settings->maxKeyPtSize0 - 1.0f)/(settings->maxKeyPtSize - settings->minKeyPtSize);
         
        keyPtsSize.push_back(keyPtSize_norm);
    }
}

void ANYFEATURE_VSLAM::FeatureExtractor::computeSigma(std::vector<mat2f>& keyPtsSigma2, std::vector<mat2f>& keyPtsInf,
                                                      const std::vector<float>& keyPtsSize, const std::vector<cv::KeyPoint>& keypoints,
                                                      const Image& img, const CovarianceMethod& method){
    keyPtsSigma2.clear();
    keyPtsInf.clear();
    keyPtsSigma2.reserve(keypoints.size());
    keyPtsInf.reserve(keypoints.size());
    switch (method) {
        case NONE:{
            for(const auto& keypoint: keypoints){
                keyPtsSigma2.emplace_back(mat2f::Identity());
                keyPtsInf.emplace_back(mat2f::Identity());
            }
            return;
        }
        case SIZE:{
            int iKeyPt{0};
            for(const auto& keypoint: keypoints){
                float keypointSigma = keyPtsSize[iKeyPt];
                float keypointSigma2 = keypointSigma * keypointSigma;
                float keypointInf = 1.0f/(keypointSigma2);
                keyPtsSigma2.emplace_back(keypointSigma2 * mat2f::Identity());
                keyPtsInf.emplace_back(keypointInf * mat2f::Identity());
                iKeyPt++;
            }
            return;
        }
    }
}

std::vector<cv::KeyPoint> ANYFEATURE_VSLAM::FeatureExtractor::DistributeOctTree(std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes   
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    std::list<ExtractorNode> lNodes;

    std::vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        cv::KeyPoint &kp = vToDistributeKeys[i];
        kp.class_id = i;
        //std::cout << kp.response << " ";
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    std::list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    std::vector<std::pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);                    
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                std::vector<std::pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(settings->maxNumFeatures);
    for(std::list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void ANYFEATURE_VSLAM::ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}