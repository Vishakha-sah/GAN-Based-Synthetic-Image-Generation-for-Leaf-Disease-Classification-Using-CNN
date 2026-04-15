# Plant Disease Classification via ResNet50, GANs & CBAM 🍃

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Computer_Vision-success)

## 📌 Project Overview
This repository contains a progressive machine learning pipeline designed to classify agricultural leaf diseases. To address the severe real-world challenge of dataset class imbalance (ranging up to 21:1), this project transitions from a baseline transfer learning approach to an advanced architecture utilizing **Generative Adversarial Networks (GANs)** for synthetic minority-class oversampling and **Convolutional Block Attention Modules (CBAM)** for targeted spatial feature extraction.

### 👥 Team Structure & My Role
This project was developed collaboratively by a team of 5 engineers as part of a university capstone. While the overall pipeline and methodology were shared, we each took ownership of specific neural network architectures to conduct a comparative analysis.

**My Specific Contributions:**
* **Architecture Ownership:** Engineered and trained the end-to-end **ResNet50** pipeline on the Pepper-Potato-Tomato subset.
* **Generative AI Implementation:** Designed and trained the **WGAN-GP** (Wasserstein GAN with Gradient Penalty) to successfully synthesize minority-class images. This specifically addressed the severely data-constrained *Potato Healthy* class (only 152 images), utilizing the Wasserstein loss to avoid the mode collapse typically seen in standard DCGANs.
* **Attention Mechanism:** Integrated the **CBAM** (Convolutional Block Attention Module) into the ResNet50 backbone, forcing the network to focus on discriminative disease lesions and boosting minority class recall to a perfect 1.00.

---

## 🛠️ Methodology & Tech Stack

1. **Baseline Classification:** Fine-tuned an ImageNet-pretrained **ResNet50** using a stratified 5-fold cross-validation protocol to establish initial benchmarks.
2. **Dynamic Data Augmentation:** Applied standard geometric and photometric transformations to combat overfitting.
3. **Synthetic Image Generation:** Deployed **WGAN-GP** and **DCGAN** to generate mathematically robust synthetic images for minority classes, evaluated via FID (Fréchet Inception Distance). 
4. **Attention Integration:** Appended sequential channel and spatial attention blocks (**CBAM**) prior to global average pooling to refine lesion localization.

**Tech Stack:** `Python`, `PyTorch`, `TorchVision`, `Scikit-learn`, `NumPy`, `Matplotlib`

---

## 🚀 Key Results & Performance
By systematically isolating variables and addressing the class imbalance on a data level, the final model demonstrated highly robust generalization on a strictly real-world, fixed test set.

* **Test Accuracy:** 99.79%
* **AUC (One-vs-Rest):** 1.0000
* **Macro F1-Score:** 0.9948
* **Generative Quality (FID Scores):** * `Pepper_bell___Bacterial_spot`: 99.83
  * `Pepper_bell___healthy`: 91.24
  * `Potato___Late_blight`: 127.72
  * `Tomato_Early_blight`: 197.13
  * `Potato___healthy`: 231.77

---

## 📂 Repository Contents
* `Vishakha_sah(23053585).ipynb`: The complete Google Colab notebook containing the model architecture, GAN training loops, CBAM implementation, and evaluation metrics.
* `Final_Report_grp_3.pdf`: The official, comprehensive academic report detailing the comparative analysis across all tested CNN architectures.
