//
// Created by fontan on 18/02/24.
//

#include "Observation.h"

AF_VSLAM::Observation::Observation(Keyframe projKeyframe, const KeypointIndex& projIndex):
        projKeyframe(projKeyframe), projIndex(projIndex){
}