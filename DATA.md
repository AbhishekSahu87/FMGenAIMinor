Of course. Here is the `DATA.md` file for this project.

***

# Data Card: KTH Action Recognition Dataset

This document provides details on the dataset used for fine-tuning and evaluating the Asymmetric Masked Distillation (AMD) model.

---
## Dataset Overview

The **KTH Action Recognition Dataset** is a classic benchmark for human action recognition. It contains videos of 25 subjects performing six different actions in a controlled environment with a static camera and uniform background. 

The six action classes are:
* boxing
* handclapping
* handwaving
* jogging
* running
* walking

---
## Provenance and Access

* **Creators**: KTH Royal Institute of Technology, Stockholm, Sweden.
* **Source Link**: The dataset can be downloaded from the official project page: **[https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)**
* **Publication**: Schuldt, C., Laptev, I., & Caputo, B. (2004). *Recognizing human actions: a local SVM approach*. In Proceedings of the 17th International Conference on Pattern Recognition (ICPR).

---
## License and Usage

The dataset is publicly available for **non-commercial, academic research purposes**. There is no explicit software-style license (like MIT or Apache), so usage should be limited to research and educational contexts as intended by the creators.

---
## Novelty Relative to the AMD Paper

The original AMD paper evaluates its model on large-scale, complex, "in-the-wild" video datasets like Kinetics-400, Something-Something V2, and AVA, which feature diverse scenes, camera motion, and clutter.

The KTH dataset is **new** in this context because it represents a significant **domain shift** from complexity to simplicity. It introduces the following new conditions:
* **Controlled Environment**: A static background and camera, unlike the dynamic scenes in Kinetics.
* **Low Complexity**: Simple, distinct actions performed by a single person, testing the model's ability to generalize to less noisy data.
* **Small Scale**: With only ~600 video clips, it tests the model's data efficiency and transfer learning capabilities from a large pre-training corpus to a small target dataset.

---
## Preprocessing and Splits

The raw AVI video files were preprocessed to prepare them for the model using a custom Python script (`prepare_kth.py`).

1.  **Splitting**: The standard KTH protocol was used to create training and testing splits.
    * **Training Set**: Videos from subjects 1-16 (384 clips).
    * **Testing Set**: Videos from subjects 17-25 (216 clips).

2.  **Annotation File Generation**: The script iterated through the video files and generated `train.txt` and `test.txt` files. Each line in these files maps a relative video path to its corresponding integer label (0-5).

3.  **Data Loading**: During training, each video was processed by:
    * Uniformly sampling 16 frames.
    * Applying data augmentation (random resized crop to 224x224 and horizontal flip).
    * Normalizing the pixel values using ImageNet's mean and standard deviation.
