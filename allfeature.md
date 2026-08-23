# AllFeature-VSLAM — `src/Tracking.cc`

## Call graph

[`LoadParameters`](src/Tracking.cc#L37), [`Tracking`](src/Tracking.cc#L56), [`loadCameraParameters`](src/Tracking.cc#L1149), [`ChangeCalibration`](src/Tracking.cc#L1116), [`get_feature_extractor`](src/Tracking_aux.cc#L91), [`grab_image`](src/Tracking.cc#L82), [`track`](src/Tracking.cc#L117), [`store_relative_pose`](src/Tracking.cc#L206), [`monocular_initialization`](src/Tracking.cc#L212), [`attempt_monocular_initialization`](src/Tracking.cc#L220), [`create_initial_map_monocular`](src/Tracking.cc#L318), [`check_replaced_in_last_frame`](src/Tracking.cc#L430), [`track_reference_keyframe`](src/Tracking.cc#L450), [`track_local_map`](src/Tracking.cc#L541), [`search_local_points`](src/Tracking.cc#L699), [`update_local_map`](src/Tracking.cc#L584), [`update_local_keyframes`](src/Tracking.cc#L593), [`update_local_points`](src/Tracking.cc#L676), [`MedianFlowFromLastFrame`](src/Tracking.cc#L855), [`need_new_keyframe`](src/Tracking.cc#L734), [`create_new_keyframe`](src/Tracking.cc#L840), [`relocalize`](src/Tracking.cc#L888), [`reset`](src/Tracking.cc#L1066)

```mermaid
flowchart TD
    RL["<b>relocalize</b><br/>· compute_global_descriptor<br/>· DetectRelocalizationCandidates<br/>· match_keyframe_to_frame<br/>· PnPsolver::iterate<br/>· PoseOptimization<br/>· SearchByProjection"]
    TRK["<b>track_reference_keyframe</b><br/>· match_keyframe_to_frame<br/>· PoseOptimization<br/>· count_inlier_map_points<br/>· dump_pose_collapse (rescue)"]
    MI["<b>monocular_initialization</b><br/>· attempt_monocular_initialization<br/>· create_initial_map_monocular"]

    T["<b>track</b><br/>· check_replaced_in_last_frame<br/>· run_tracking_stage"] -->|state == LOST| RL
    T -->|state == OK| TRK
    T -->|NOT_INITIALIZED| MI

    RL -->|lost| NF1["next frame → relocalize"]
    RL -->|ok| TLM["<b>track_local_map</b><br/>· update_local_map → update_local_keyframes, update_local_points<br/>· search_local_points → match_map_points_to_frame<br/>· PoseOptimization"]
    TRK -->|ok| TLM
    TRK -->|lost| NF2["next frame → relocalize"]
    TLM -->|lost| NF3["next frame → relocalize"]
    TLM -->|ok| KF["<b>need_new_keyframe / create_new_keyframe</b><br/>· MedianFlowFromLastFrame<br/>· insert_keyframe"]
    KF --> TE["<b>store_relative_pose</b>"]
    TE -->|emergency_keyframe_| EW["<b>emergency keyframe wait</b><br/>· unlock map mutex<br/>· spin until local_mapper_->accepts_keyframes()"]
    MI -->|not initialized| RET[return — retry next frame]
    MI -->|map created| TE
    MI ~~~ TLM
```
