# AllFeature-VSLAM — code reference

Index of the per-file reference pages under [`docs/reference/`](docs/reference/) (format:
[`docs/reference/README.md`](docs/reference/README.md)). The same pages, with clickable diagrams,
are served by the documentation site built from `docs/` (see `mkdocs.yml`). The author's reading
notes are in [`docs/review/`](docs/review/), measured changes in [`docs/gym/`](docs/gym/), dated
investigations in [`docs/notes/`](docs/notes/), plans in [`docs/plans/`](docs/plans/).

## System

```mermaid
flowchart TD
    CLI(["vslamlab_allfeature_mono / _rgbd<br/>· settings YAML · rgb.csv · calibration"]) --> SYS["<b>System</b><br/>· owns Map, PlaceRecognition<br/>· starts the threads · saves outputs"]
    SYS --> TRK["<b>Tracking</b> (caller's thread)<br/>· extract features (+ mask)<br/>· initialize · track · relocalize<br/>· keyframe decision"]
    SYS --> LM["<b>LocalMapping</b> (thread)<br/>· new map points (triangulated + depth-seeded)<br/>· map-point culling · local BA<br/>· keyframe culling"]
    SYS --> LC["<b>LoopClosing</b> (thread, vpr ≠ none)<br/>· detect loop · Sim3 · correct · global BA"]
    SYS --> VW["<b>Viewer</b> (thread, verbose:1)<br/>· Pangolin map + frame · placecell window"]
    TRK -- "insert_keyframe" --> LM
    LM -- "insert_keyframe" --> LC
    TRK <--> MAP[("Map<br/>KeyFrame · MapPoint")]
    LM <--> MAP
    LC <--> MAP
    VW -.-> MAP
    TRK <--> VPR[("PlaceRecognition<br/>megaloc | none<br/>(placecell)")]
    LM <--> VPR
    LC <--> VPR
    TRK --> OPT["Optimizer (g2o)<br/>pose · local BA · Sim3 · essential graph · global BA"]
    LM --> OPT
    LC --> OPT
    TRK --> FM["Feature · FeatureExtractor · FeatureMatcher<br/>orb32 … superpoint256 · BF / LightGlue · PoseLib filter"]
    LM --> FM
    LC --> FM

    click SYS "docs/reference/System.md"
    click TRK "docs/reference/Tracking.md"
    click LM "docs/reference/LocalMapping.md"
    click LC "docs/reference/LoopClosing.md"
    click VW "docs/reference/Viewer.md"

    %% VSLAM-LAB logo squares: cyan #b5f3f9, periwinkle #8195fb, lavender #a59ddf
    classDef entry fill:#8195fb,stroke:#5f74d6,color:#fff
    classDef step fill:#b5f3f9,stroke:#7fcfd8,color:#1b2a4a
    classDef check fill:#a59ddf,stroke:#7e75c4,color:#1b2a4a
    classDef cmd fill:#fff,stroke:#8195fb,color:#1b2a4a
    classDef store fill:#fff,stroke:#a59ddf,color:#1b2a4a

    class CLI entry
    class SYS,TRK,LM,LC,VW step
    class OPT,FM cmd
    class MAP,VPR store
```

## Pages

| Page | Sources | Thread |
|---|---|---|
| [System](docs/reference/System.md) | `src/System.cc`, `include/System.h` | caller's |
| [Tracking](docs/reference/Tracking.md) | `src/Tracking.cc`, `src/Tracking_aux.cc`, `include/Tracking.h` | caller's (per frame) |
| [LocalMapping](docs/reference/LocalMapping.md) | `src/LocalMapping.cc`, `src/LocalMapping_aux.cc`, `include/LocalMapping.h` | local-mapping |
| [LoopClosing](docs/reference/LoopClosing.md) | `src/LoopClosing.cc`, `src/LoopClosing_aux.cc`, `include/LoopClosing.h` | loop-closing (+ GBA thread) |
| [Viewer](docs/reference/Viewer.md) | `src/Viewer.cc`, `include/Viewer.h` (+ `FrameDrawer`, `MapDrawer`) | viewer |

Components without a page yet (next in line, one per review session): `Frame` / `KeyFrame` /
`MapPoint` / `Map`, `FeatureExtractor` / `FeatureMatcher` / `BruteForceMatcher`, `Optimizer`,
`PlaceRecognition` / `PlaceRecognitionMegaLoc`, `Initializer`, `PnPsolver` / `Sim3Solver`,
`Image` / segmentation, the CLI entry points.
