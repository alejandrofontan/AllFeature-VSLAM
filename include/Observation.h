//
// Created by fontan on 18/02/24.
//

#ifndef AF_VSLAM_OBSERVATION_H
#define AF_VSLAM_OBSERVATION_H

#include "KeyFrame.h"
#include "FeatureFactory.h"

namespace AF_VSLAM {

    class KeyFrame;

    class Observation {
    public:
        Observation() = default;

        Observation(Keyframe projKeyframe, const KeypointIndex &projIndex);

        // The keyframe observing the map point and the index of its keypoint there
        Keyframe projKeyframe{};
        KeypointIndex projIndex{};
    };

    typedef shared_ptr<Observation> Obs;

}
#endif //AF_VSLAM_OBSERVATION_H
