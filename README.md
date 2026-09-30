# Breast Cancer Classification Using Deep Learning and Radiomics

A medical imaging framework for breast cancer classification from mammography images using convolutional neural networks, handcrafted radiomics features, machine learning, and multimodal fusion.

---

## Overview

Breast cancer diagnosis from mammography remains challenging because malignant and benign lesions can present with similar visual characteristics.

This project investigates whether combining **deep learning representations** with **handcrafted radiomics features** can provide complementary information for breast lesion classification.

The framework evaluates multiple convolutional neural network architectures together with radiomics-based machine learning and fusion strategies.

The major components include:

1. Mammography image preprocessing.
2. Region-of-interest and full-image analysis.
3. Deep learning classification using multiple CNN architectures.
4. Radiomics feature extraction using PyRadiomics.
5. Machine-learning classification using XGBoost.
6. Deep learning and radiomics fusion.
7. Model calibration and decision analysis.
8. Visual interpretation using Grad-CAM++.

---

## Research Objective

The primary objective of this project is to investigate different representations of mammography images for binary breast lesion classification.

The study compares:

- CNN-based image representations.
- Handcrafted radiomics features.
- Machine-learning classification.
- Prediction-level fusion.
- Feature-level / joint fusion.
- Model interpretability and clinical utility.

---

## Dataset

This project uses mammography data derived from the **CBIS-DDSM** dataset.

The dataset was divided at the patient level to reduce information leakage between training, validation, and testing subsets.

| Split | Number of Images |
|---|---:|
| Training | 968 |
| Validation | 246 |
| Testing | 341 |

The test set contains:

| Class | Number of Samples |
|---|---:|
| Class 0 | 194 |
| Class 1 | 147 |
| **Total** | **341** |

Patient-wise splitting was used so that images belonging to the same patient do not appear across different data partitions.

---

## System Pipeline

```text
                     Mammography Image
                            │
                            ▼
                       Preprocessing
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
              ROI Image          Full Image
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
                     CNN Classification
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
      ResNet18         DenseNet121         GoogLeNet
                                                │
                                                ▼
                                           SqueezeNet
                            │
                            ▼
                    Deep Model Prediction
                            │
                            │
          Mammography ROI   │
                │           │
                ▼           │
       Radiomics Extraction │
          using PyRadiomics │
                │           │
                ▼           │
         Radiomics Features │
                │           │
                ▼           │
             XGBoost        │
                │           │
                ▼           ▼
        Radiomics Prediction
                │
                └───────────┐
                            ▼
                    Multimodal Fusion
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Prediction Fusion            Joint Fusion
              │                           │
              └─────────────┬─────────────┘
                            ▼
               Breast Cancer Classification
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          ROC Analysis   Calibration   Decision Curve
                                            Analysis
                            │
                            ▼
                      Grad-CAM++
                     Visualization
```

---

## Deep Learning Models

Four convolutional neural network architectures were investigated.

### ResNet18

ResNet18 uses residual connections to improve gradient propagation and facilitate the training of deeper neural networks.

### DenseNet121

DenseNet121 connects each layer to subsequent layers, encouraging feature reuse and efficient information flow.

### GoogLeNet

GoogLeNet uses inception modules to capture visual information at multiple receptive-field scales.

### SqueezeNet

SqueezeNet provides a lightweight CNN architecture with fewer parameters while retaining meaningful image representations.

The models were evaluated using both region-of-interest and full-image mammography inputs where applicable.

---

## Radiomics Analysis

Radiomics provides quantitative handcrafted features describing lesion characteristics that may not be directly represented by standard image intensity analysis.

Radiomics features were extracted using:

```text
PyRadiomics
```

A total of approximately:

```text
118 radiomics features
```

were used in the radiomics-based analysis.

These features represent quantitative characteristics related to lesion intensity, texture, and structural properties.

The extracted features were used to train an **XGBoost classifier**.

---

## XGBoost Model

The radiomics classification component uses XGBoost.

The experimental configuration included:

```text
Maximum tree depth : 4
Number of estimators: 800
Learning rate       : 0.05
```

The radiomics model achieved a test ROC-AUC of approximately:

```text
0.5914
```

This experiment provides a baseline for evaluating whether handcrafted radiomics features offer complementary information to deep CNN representations.

---

## Fusion Strategies

One of the main objectives of this project is to investigate whether combining deep learning and radiomics can improve classification performance.

### Prediction-Level Fusion

Predictions generated independently by the CNN and radiomics models can be combined at the probability level.

A simple formulation is:

```text
P_final = α × P_CNN + (1 - α) × P_Radiomics
```

where:

- `P_CNN` represents the CNN probability.
- `P_Radiomics` represents the radiomics model probability.
- `α` controls the contribution of each model.

### Joint / Feature-Level Fusion

Deep representations obtained from CNN models can also be combined with radiomics features before final classification.

Conceptually:

```text
CNN Features
      │
      ├──────────────┐
      │              │
      ▼              ▼
Deep Representation Radiomics Features
      │              │
      └──────┬───────┘
             ▼
      Feature Concatenation
             │
             ▼
        Classification
```

This approach investigates whether learned and handcrafted representations provide complementary information.

---

## Evaluation Metrics

Model performance can be evaluated using several complementary metrics.

The primary evaluation measures include:

```text
Accuracy
Precision
Recall / Sensitivity
Specificity
F1-score
ROC-AUC
Confusion Matrix
```

ROC-AUC is particularly useful for measuring discrimination performance independently of a single classification threshold.

---

## ROC Analysis

Receiver Operating Characteristic curves are used to examine the trade-off between sensitivity and false-positive rate across classification thresholds.

The Area Under the ROC Curve provides an overall measure of the model's discriminatory ability.

```text
Higher ROC-AUC → Better discrimination
ROC-AUC = 0.5 → Performance similar to random classification
ROC-AUC = 1.0 → Perfect discrimination
```

---

## Model Calibration

Good discrimination does not necessarily imply that predicted probabilities are well calibrated.

Therefore, calibration analysis is included to assess whether predicted probabilities correspond to observed outcome frequencies.

Calibration analysis can include:

```text
Calibration Curve
Brier Score
Predicted Probability Distribution
```

This is particularly important for medical decision-support applications where probability estimates may influence clinical decisions.

---

## Decision Curve Analysis

Decision Curve Analysis (DCA) is used to investigate the potential clinical usefulness of a prediction model across different decision thresholds.

Rather than considering classification performance alone, DCA evaluates the relationship between:

```text
True-positive benefit
        versus
False-positive harm
```

across clinically relevant threshold probabilities.

---

## Explainable AI

To improve model interpretability, this project includes **Grad-CAM++** visualization.

Grad-CAM++ generates class-specific activation maps that highlight image regions contributing strongly to a CNN prediction.

Conceptually:

```text
Mammography Image
       │
       ▼
      CNN
       │
       ▼
Predicted Class
       │
       ▼
 Grad-CAM++
       │
       ▼
Activation Heatmap
```

These visualizations can help determine whether the model is focusing on clinically meaningful lesion regions.

> Grad-CAM++ should be interpreted as a model-explanation technique rather than direct evidence of clinical reasoning.

---

## Repository Structure

```text
breast-cancer-radiomics-deep-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── preprocess_images.py
│   │   └── prepare_dataset.py
│   │
│   ├── cnn/
│   │   ├── train_resnet.py
│   │   ├── train_densenet.py
│   │   ├── train_googlenet.py
│   │   ├── train_squeezenet.py
│   │   └── inference.py
│   │
│   ├── radiomics/
│   │   ├── extract_radiomics.py
│   │   └── train_xgboost.py
│   │
│   ├── fusion/
│   │   ├── late_fusion.py
│   │   └── joint_fusion.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── roc_analysis.py
│   │   ├── calibration.py
│   │   └── decision_curve.py
│   │
│   └── explainability/
│       └── gradcam.py
│
├── configs/
│   └── experiment_config.yaml
│
├── scripts/
│   ├── prepare_data.py
│   └── run_experiments.py
│
└── assets/
    ├── system_pipeline.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── calibration_curve.png
    ├── decision_curve.png
    └── gradcam_example.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/najeebullahr/breast-cancer-radiomics-deep-learning.git
```

Move into the project directory:

```bash
cd breast-cancer-radiomics-deep-learning
```

Create a virtual environment if desired:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

The project is primarily developed using:

```text
Python
PyTorch
TorchVision
NumPy
Pandas
OpenCV
Scikit-learn
XGBoost
PyRadiomics
SimpleITK
Matplotlib
```

Exact package versions will be documented according to the experimental environment used for reproducibility.

---

## Example Workflow

The overall experimental workflow is:

```text
1. Prepare mammography images
        ↓
2. Generate patient-wise train/validation/test splits
        ↓
3. Train CNN models
        ↓
4. Extract radiomics features
        ↓
5. Train XGBoost radiomics classifier
        ↓
6. Evaluate individual models
        ↓
7. Perform deep learning + radiomics fusion
        ↓
8. Evaluate ROC-AUC and classification metrics
        ↓
9. Assess calibration
        ↓
10. Perform Decision Curve Analysis
        ↓
11. Generate Grad-CAM++ visualizations
```

---

## Data Availability

The original mammography images are **not redistributed through this repository**.

Users interested in reproducing the experiments should obtain the CBIS-DDSM dataset from its authorized distribution source and comply with the corresponding dataset terms and conditions.

The repository is intended to provide the methodology, implementation, evaluation pipeline, and reproducibility resources rather than redistribute medical imaging data.

---

## Medical and Research Disclaimer

This project is intended for academic and research purposes.

The models and outputs provided through this repository are not intended to replace professional medical diagnosis or clinical decision-making.

Clinical deployment would require additional validation, regulatory assessment, external testing, and evaluation across diverse patient populations.

---

## Limitations

Several factors should be considered when interpreting the experimental results.

The dataset size is limited compared with large-scale clinical imaging datasets, and performance may vary across imaging devices, institutions, patient populations, and acquisition protocols.

Radiomics features may also be sensitive to preprocessing, segmentation, image resolution, and feature-extraction settings.

Further external validation is therefore necessary before conclusions regarding clinical generalizability can be made.

---

## Future Work

Future development of this project may include:

- External validation using independent mammography datasets.
- Larger patient cohorts.
- Improved lesion segmentation.
- Feature-selection strategies for radiomics.
- Advanced multimodal fusion.
- Transformer-based image models.
- Model uncertainty estimation.
- Improved probability calibration.
- Explainability comparison across CNN architectures.
- Clinical validation of decision-support performance.

---

## Project Status

The project includes completed experiments involving:

```text
CNN classification
Radiomics analysis
XGBoost classification
Deep learning–radiomics fusion
ROC analysis
Model calibration
Grad-CAM++ visualization
Decision Curve Analysis
```

The public repository is being organized to provide a clean and reproducible implementation of the research workflow.

---

## Author

**Najeeb Ullah**

Master's Student  
National Taipei University

Research interests:

```text
Medical Artificial Intelligence
Medical Image Analysis
Deep Learning
Computer Vision
Radiomics
Healthcare Machine Learning
Medical NLP
```

### Connect

[LinkedIn](https://www.linkedin.com/in/najeebullahr)

---

## Acknowledgment

This repository is developed for academic research in medical artificial intelligence and breast imaging analysis.

---

## Citation

If this repository contributes to your research, please cite the corresponding publication when it becomes available.

```bibtex
@misc{ullah_breast_cancer_ai,
  author       = {Najeeb Ullah},
  title        = {Breast Cancer Classification Using Deep Learning and Radiomics},
  year         = {2026},
  howpublished = {GitHub repository}
}
```
