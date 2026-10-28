# 🦷 Oral Disease Image Classification

Deep learning project that classifies oral images into **six conditions**: Calculus, Caries, Gingivitis, Hypodontia, Mouth Ulcer and Tooth Discoloration. A custom CNN is compared against transfer-learning models (MobileNetV2, EfficientNetB0), and the best model is deployed with Gradio.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahmoudeimo95/oral-disease-classification/blob/main/oral_disease_classification.ipynb)

> ⚠️ Educational project only. Not a medical diagnosis tool.

## Dataset

[Oral Diseases (Kaggle)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases): 5,563 images after using the original (non-augmented) folders for Caries, Mouth Ulcer and Tooth Discoloration, so near-duplicate augmented copies cannot leak across splits.

| Class | Images |
|---|---|
| Gingivitis | 2,349 |
| Calculus | 1,296 |
| Hypodontia | 1,251 |
| Mouth Ulcer | 265 |
| Caries | 219 |
| Tooth Discoloration | 183 |

The data is imbalanced, so class weights are used during training. A stratified **70 / 15 / 15** split gives 3,894 train / 834 validation / 835 test images.

## Pipeline

1. Data loading and exploration
2. Preprocessing: resize to 224×224, `tf.data` pipeline, augmentation (flip, rotation, zoom, brightness, contrast) applied only during training
3. Custom CNN baseline (VGG-style, trained from scratch)
4. Transfer learning with frozen MobileNetV2 and EfficientNetB0
5. Hyperparameter tuning (learning rate × dropout, selected by validation accuracy) and fine-tuning of the last 30 backbone layers
6. Comparative analysis and final evaluation
7. Export and Gradio deployment

## Results (test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 0.60 | 0.53 |
| MobileNetV2 (frozen) | 0.76 | 0.73 |
| EfficientNetB0 (frozen) | 0.75 | 0.72 |
| MobileNetV2 (tuned, LR 1e-3, dropout 0.5) | 0.75 | 0.72 |
| **MobileNetV2 (fine-tuned), final model** | high 0.70s | ≈ 0.77 |

The fine-tuned MobileNetV2 was selected as the final model: best accuracy and macro F1, and a small file (~22 MB). The main confusion is between **Calculus and Gingivitis**, which look similar and often co-occur; the small classes (Caries, Tooth Discoloration) are the least stable.

## Repository structure

```
├── oral_disease_classification.ipynb   # full pipeline (Colab, T4 GPU)
├── app.py                              # Gradio app
├── final_model.keras                   # trained model used by the app
├── requirements.txt
├── LICENSE
└── README.md
```

## Run the app locally

```bash
git clone https://github.com/mahmoudeimo95/oral-disease-classification.git
cd oral-disease-classification
pip install -r requirements.txt
python app.py
```

Then open the local URL printed in the terminal.

## Reproduce the training

Click the **Open in Colab** badge above, set the runtime to **GPU (T4)**, and run all cells. The dataset is downloaded automatically with `kagglehub`.

## Tech stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Gradio

## Author

**Mahmoud**: [GitHub](https://github.com/mahmoudeimo95) · [Freelancer](https://www.freelancer.com/u/mahmoudm301)
