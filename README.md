[comment]: <> (# AllFeature-VSLAM)

<p align="center">
<div align="center">
    <img src="wisdom_kookaburra.png" width="500"/>
</div>

<p align="center">
  </h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=SDtnGogAAAAJ&hl=en&oi=a"><strong>Alejandro Fontan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en&oi=a"><strong>Javier Civera</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en&oi=ao"><strong>Michael Milford</strong></a>
  </p>

## Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
*After installation, restart your terminal or source your shell for the changes to take effect*. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

*If you already have pixi remember to update:* `pixi self-update`

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/alejandrofontan/AllFeature-VSLAM.git --recursive && cd AllFeature-VSLAM
```
Build with
```bash
pixi run build
```
Run demo
```bash
pixi run demo
```


## License
**AllFeature-VSLAM** builds on [**ORB-SLAM2**](https://github.com/raulmur/ORB_SLAM2) and inherits its release under a [GPLv3 license](https://github.com/alejandrofontan/AnyFeature-VSLAM/blob/main/docs/License-gpl.txt). For a list of all other code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/alejandrofontan/AnyFeature-VSLAM/blob/main/docs/Dependencies.md).

**Acknowledgments to:** [Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/), [J. M. M. Montiel](http://webdiis.unizar.es/~josemari/) ([ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)), [Dorian Galvez-Lopez](http://doriangalvez.com/) ([DBoW2](https://github.com/dorian3d/DBoW2)), Carlos Campos, Richard Elvira and Juan J. Gómez Rodríguez ([ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)).

## Related Publications:
[AnyFeature-VSLAM] Alejandro Fontan, Javier Civera and Michael Milford, **Automating the Usage of Any Chosen Feature into Visual SLAM**, *Robotics: Science and Systems, 2024*. **[PDF](https://roboticsconference.org/program/papers/84/)**.

[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. **[PDF](https://arxiv.org/abs/2007.11898)**.

[ORB-SLAM2] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://128.84.21.199/pdf/1610.06475.pdf)**.

[ORB-SLAM] Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. *IEEE Transactions on Robotics,* vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)**.

[DBoW2 Place Recognizer] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp.  1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**
