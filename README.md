# Spine X-ray Sex and BMI Prediction

This project develops a multi-task computer vision system that predicts a sex
label and body mass index (BMI) from spine X-ray images. It compares convolutional
neural networks trained from scratch with pretrained ResNet18 and EfficientNet-B0
models using the PyTorch ecosystem.

## Project Overview

The notebook covers:

- Exploratory analysis of image properties, label distributions, and possible imbalance
- Image resizing, normalization, and data augmentation
- A custom multi-task PyTorch dataset and data loaders
- Joint classification and regression with separate prediction heads
- Twelve systematic model and training experiments
- Comparison of CNN, ResNet18, and EfficientNet-B0 architectures
- Error analysis, subgroup evaluation, and confusion-matrix analysis
- Grad-CAM visualizations for model explainability
- Generation of a Kaggle submission file

## Dataset

The project uses the private dataset supplied through the `bucv26` Kaggle
competition. The notebook expects:

```text
train_labels.csv
sample_submission.csv
train_images/
test_images/
```

The dataset contains 1,000 training images and 195 test images in the notebook's
recorded run. Competition data is not included in this repository and remains
subject to Kaggle's competition rules and access restrictions.

## Multi-task Learning

Each model shares an image-feature backbone and produces two outputs:

- **Sex classification:** optimized with cross-entropy loss
- **BMI regression:** optimized with Smooth L1 loss

The combined training objective is the sum of the classification and regression
losses.

## Experiments

The notebook evaluates:

- Baseline and deeper CNN architectures trained from scratch
- Data augmentation
- Dropout regularization
- Adam and AdamW optimizers
- Pretrained ResNet18
- Pretrained EfficientNet-B0
- Extended training

The selected final model is a pretrained **ResNet18 with AdamW**.

## Recorded Results

Final-model validation results from the notebook:

| Metric | Value |
|---|---:|
| Sex classification accuracy | 0.880 |
| Precision | 0.960 |
| Recall | 0.686 |
| F1 score | 0.800 |

The recorded Kaggle private leaderboard score is **0.5356**.

Subgroup analysis showed a notable performance difference:

| Validation group | Accuracy | Samples |
|---|---:|---:|
| Sex label 0 | 0.985 | 130 |
| Sex label 1 | 0.686 | 70 |

This difference indicates possible class imbalance, dataset bias, or unequal model
performance and should be considered when interpreting the aggregate metrics.

## Technologies

- Python
- PyTorch and Torchvision
- pandas and NumPy
- scikit-learn
- Matplotlib
- Pillow
- Grad-CAM
- Kaggle API
- Google Colab

## Running the Notebook

1. Open `Spine_Xray_Sex_BMI_Prediction.ipynb` in Google Colab.
2. In Colab's **Secrets** panel, add:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
3. Enable notebook access for both secrets.
4. Confirm that your Kaggle account has access to the `bucv26` competition.
5. Run the cells in order. A GPU runtime is recommended.

The notebook retrieves Kaggle credentials securely:

```python
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
```

Never commit Kaggle credentials, API keys, generated credential files, competition
data, or restricted model artifacts to Git.

## Repository Structure

```text
spine-xray-sex-bmi-prediction/
├── Spine_Xray_Sex_BMI_Prediction.ipynb
└── README.md
```

## Limitations and Responsible Use

This project is for **educational and research purposes only**. It is not a
validated medical device and must not be used for diagnosis, clinical decisions,
patient assessment, or real-world inference about individuals.

The dataset is small, the labels and population may not be representative, and
the recorded subgroup performance is unequal. Sex labels may also simplify or
misrepresent biological and social variation. Results are based on a single
coursework dataset and validation setup and do not establish clinical validity or
generalization to other populations, imaging devices, or acquisition protocols.

