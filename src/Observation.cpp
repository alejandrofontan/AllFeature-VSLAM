//
// Created by fontan on 18/02/24.
//

#include "Observation.h"

AF_VSLAM::Observation::Observation(Keyframe projKeyframe, const KeypointIndex& projIndex,
                                    Keyframe ref_keyframe,  const KeypointIndex& refIndex):
        projKeyframe(projKeyframe), projIndex(projIndex),
        ref_keyframe(ref_keyframe), refIndex(refIndex){
}