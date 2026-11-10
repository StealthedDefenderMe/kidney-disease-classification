# 🩺 Kidney Disease Classification Using Deep Learning

<p align="center">
  <b>A Deep Learning Project using Transfer Learning and VGG16</b>
  <br/>
  Classifying Kidney CT Scan Images into Normal, Cyst, Tumor and Stone.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Model-VGG16-green" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-red" />
</p>

---

## 📌 Project Overview

This project uses **Deep Learning and Transfer Learning** to classify kidney CT scan images into four different categories.

Instead of training a CNN from scratch, we use a pretrained CNN architecture called **VGG16**, which has already learned useful visual patterns from the ImageNet dataset.

We reuse its feature extraction layers and add our own Dense layers to classify kidney images.

### 🎯 Classification Categories

| Class | Description |
|:---:|---|
| Normal | Healthy kidney |
| Cyst | Kidney cyst |
| Tumor | Kidney tumor |
| Stone | Kidney stone |

---

## 🏗️ Project Setup

### Step 1: Create Project Structure

Create a `template.py` file to automatically generate the required folder structure after cloning the repository.

### Step 2: Configure Setup

- Create `setup.py` to configure the project as a Python package.
- Create a virtual environment.
- Activate the virtual environment.
- Add project dependencies to `requirements.txt`.

### Step 3: Logging and Exception Handling

- Implement a centralized logging module.
- Configure custom exception handling.
- Initialize these utilities inside `__init__.py` so they are automatically available when the package is imported.

### Step 4: Common Utilities

Create `src/cnn_classification/utils/common.py` to maintain reusable functions and common application utilities.

### Step 5: Main Entry Point

Import and execute the required pipelines through `main.py`.

---

## 🔄 Project Workflow

The project follows a modular architecture consisting of configuration, entity, component and pipeline layers.

| Step | File / Layer | Responsibility |
|:---:|---|---|
| 1 | `config.yaml` | Define project configurations and paths |
| 2 | `secrets.yaml` | Maintain sensitive configurations (Optional) |
| 3 | `params.yaml` | Define model hyperparameters |
| 4 | Entity | Define configuration data structures |
| 5 | Configuration Manager | Load configurations and create entity objects |
| 6 | Components | Implement individual ML operations |
| 7 | Pipelines | Connect components into workflows |
| 8 | `main.py` | Execute the required pipelines |
| 9 | `dvc.yaml` | Track the complete ML pipeline |
| 10 | `app.py` | Integrate the trained model into an application |

### Component Structure

The components contain the actual implementation logic.

- **Data Injection:** Download and store the dataset.
- **Data Transformation:** Resize, preprocess and prepare images.
- **Model Preparation:** Download and prepare the pretrained VGG16 model.
- **Model Training:** Train and fine-tune the model.
- **Model Evaluation:** Evaluate model performance.

---

## 📥 1. Data Injection

Data injection is the first component of the machine learning pipeline.

Its responsibility is to download the kidney CT scan dataset and store it in the project's artifact directory.

**Dataset:** Kidney CT Scan Dataset

The dataset contains four classes:

- Normal
- Cyst
- Tumor
- Stone

---

## 🧠 2. Model Preparation – Transfer Learning

Instead of building a CNN from scratch, we use **VGG16**, a pretrained Convolutional Neural Network.

### What is a Pretrained CNN?

A CNN (Convolutional Neural Network) is a deep learning architecture designed to understand images.

Pretrained means someone has already trained the model on a large image dataset like ImageNet.

Instead of starting from zero:

```text
Random CNN ❌
     |
     v
Learn everything from scratch
```

We start with:

```text
Pretrained CNN ✅
     |
     v
Already knows many visual patterns
```

### Model Architecture

<p align="center">
  <b>VGG16 + Custom Classification Layers</b>
</p>

```text
             Kidney CT Image
                    |
                    v
              Pretrained VGG16
                    |
                    v
          CNN Feature Extraction Layers
                    |
                    v
          GlobalAveragePooling2D
                    |
                    v
               Dense (128)
                    |
                    v
             Dense (4, Softmax)
                    |
                    v
             Final Prediction
                    |
          +---------+---------+
          |         |         |
        Normal     Cyst      Tumor      Stone
```

### Understanding the Architecture

#### 1. Pretrained CNN – VGG16

VGG16 is a CNN architecture that has already learned useful visual patterns from ImageNet.

We reuse its convolutional layers instead of learning everything from scratch.

#### 2. CNN Feature Extraction Layers

Inside VGG16, different convolutional layers progressively learn different visual patterns.

- **Early layers:** Edges and lines.
- **Middle layers:** Shapes and textures.
- **Deeper layers:** Complex patterns and structures.

We keep these layers because their learned features can also help understand kidney CT images.

#### 3. GlobalAveragePooling2D (GAP)

VGG16 produces 512 feature maps in our architecture.

Each feature map represents how strongly a particular learned visual pattern appears at different spatial locations.

GAP calculates the average value of each feature map, converting it into one number.

**Example:**

Imagine one feature map contains:

```text
+---+---+
| 2 | 4 |
+---+---+
| 6 | 8 |
+---+---+
```

GAP calculates its average:

\[
\frac{2+4+6+8}{4}=5
\]

Therefore, this feature map produces the number `5`.

GAP repeats this operation for all 512 feature maps.

```text
512 Feature Maps
       |
       v
GlobalAveragePooling2D
       |
       v
512 Numbers
```

#### 4. Dense / ANN Layers

Now we add our own classification layers.

| Layer | Responsibility |
|---|---|
| `Dense(128, relu)` | Learns useful combinations of extracted features |
| `Dense(4, softmax)` | Produces probabilities for the four classes |

The Softmax layer produces four probabilities:

- Normal
- Cyst
- Tumor
- Stone

The class with the highest probability becomes the model's prediction.

<details>
<summary><b>📖 Complete Model Flow – Simple Explanation</b></summary>

```text
Kidney CT Image
      |
      v
     VGG16
      |
      v
Extract Visual Features
      |
      v
Global Average Pooling
      |
      v
   512 Numbers
      |
      v
   Dense (128)
      |
      v
 Dense (4, Softmax)
      |
      v
Normal / Cyst / Tumor / Stone
```

In simple words:

1. VGG16 understands the visual patterns in the image.
2. GAP converts those patterns into 512 numbers.
3. Dense layers learn how to use those numbers for classification.
4. Softmax predicts the final class.

</details>

---

## 🖼️ 3. Data Transformation

The transformation component prepares raw images before feeding them into the model.

### Current Transformation Process

1. Read images from the `Normal`, `Cyst`, `Tumor`, and `Stone` folders.
2. Resize every image to `224 × 224`.
3. Automatically generate labels based on folder names.
4. Create batches containing 16 images.
5. Split the dataset into:
   - 80% → Training
   - 20% → Validation
6. Use a fixed random seed (`123`) to make the split reproducible.

### Next Stage of Transformation

The following operations help improve the training process:

| Operation | Purpose |
|---|---|
| Data Augmentation | Introduce variations in training images |
| VGG16 Preprocessing | Prepare images according to VGG16's expected input format |
| Caching | Avoid repeatedly loading and processing the same images |
| Prefetching | Prepare the next batch while the model processes the current batch |

> [!NOTE]
> Data augmentation is applied only to training images. Validation images remain unchanged.

---

## 🏋️ 4. Model Training

Once the data transformation is complete, the training component loads the prepared VGG16 model and trains it using the training dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | VGG16 |
| Image Size | 224 × 224 × 3 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Output Classes | 4 |

### Fine-Tuning

Initially, the pretrained VGG16 layers are frozen so that their existing learned features are preserved.

Later, we unfreeze the last four layers to allow the model to fine-tune some of its learned features for kidney CT images.

### Model Checkpoint

We use `ModelCheckpoint` to save the best-performing model based on validation loss.

This allows us to retain the best model rather than simply using the model from the final epoch.

---

## 📊 5. Model Evaluation

Model evaluation helps us understand how well our trained model performs on unseen images.

The evaluation stage focuses on:

- Accuracy
- Loss
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

The final goal is to evaluate the model on a separate test dataset that was not used during training or validation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| VGG16 | Pretrained CNN architecture |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Data splitting and evaluation |
| DVC | Machine learning pipeline tracking |
| FastAPI | Prediction API |

---

## 🎯 Project Goal

Build a modular and scalable deep learning application that can classify kidney CT scan images into four categories using transfer learning.

The project also aims to support future model improvements through incremental training and integration with a prediction API.

---

> [!IMPORTANT]
> This project is intended for educational and research purposes. Its predictions are not a substitute for professional medical diagnosis.

---

## 🧩 6. Understanding How the Model Learns

Your model has two main parts:

### Model Architecture – Feature Extractor vs Classification Head

| Layer | Purpose |
|---|---|
| VGG16 Convolutional Layers | Extract visual features from kidney images |
| Global Average Pooling | Summarizes feature maps into 512 numbers |
| Dense(128, ReLU) | Learns useful combinations of extracted features |
| Dense(4, Softmax) | Predicts Cyst, Normal, Stone, or Tumor |

### Complete Learning Flow

```text
         Kidney CT Image
                |
                v
           VGG16 CNN
         (Feature Extractor)
                |
                v
      Global Average Pooling
          (512 Numbers)
                |
                v
          Dense (128, ReLU)
       (Learns Feature Patterns)
                |
                v
         Dense (4, Softmax)
                |
                v
        Final Classification
```

> [!NOTE]
> VGG16 itself contains the CNN convolutional layers, while Global Average Pooling and the two Dense layers form the classification head we added.

### How Does the Model Learn?

Imagine your model receives a kidney image whose actual label is **Tumor**.

The model predicts:

| Class | Predicted Probability |
|---|---:|
| Cyst | 70% |
| Tumor | 20% |
| Normal | 5% |
| Stone | 5% |

The model is wrong because the actual class is Tumor, but it predicted Cyst.

This is how the training process corrects such mistakes:

1. **Prediction:** The model predicts probabilities for all four classes.
2. **Calculate Loss:** The loss function measures how far the prediction is from the correct label.
3. **Backpropagation:** The model calculates how its trainable weights contributed to the error.
4. **Update Weights:** The optimizer (Adam in our project) adjusts those weights to reduce the loss.
5. **Repeat:** The model processes more images and repeats this process over multiple epochs.

### What Happens to VGG16 During Training?

In our project:

- Most VGG16 layers are frozen.
- The last four VGG16 layers are unfrozen.
- Our newly added Dense layers are trainable.

This means the Dense layers learn to classify kidney images, while the unfrozen VGG16 layers can also adapt their learned features to our dataset.

> [!IMPORTANT]
> Training aims to reduce errors, but improvement on training images alone doesn't guarantee better predictions on unseen images. That's why we monitor validation performance.

---

## 🔍 7. Model Explainability Using Grad-CAM

### Why Do We Need Grad-CAM?

Our model can predict whether a kidney CT image belongs to Cyst, Normal, Stone, or Tumor.

But there's another important question:

**Why did the model make that prediction?**

To investigate this, we'll use **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM generates a heatmap that highlights image regions that contributed most to a particular class prediction.

### Understanding Grad-CAM with an Example

Imagine you ask your friend:

"Is this kidney CT image showing a cyst or a tumor?"

Your model predicts:

<blockquote>
  **Predicted Class:** Cyst

  **Confidence:** 99.56%

  **Actual Class:** Tumor
</blockquote>

Now we want to investigate:

**Why did the model think it was a cyst?**

That's where Grad-CAM helps.

### How Grad-CAM Works

1. **Give the Model an Image**

   The kidney CT image is passed through our trained VGG16 model.

2. **Select the Class to Investigate**

   We select the Cyst class score to investigate why the model predicted Cyst.

3. **Inspect the Final Convolutional Layer**

   We'll use `block5_conv3`, the final convolutional layer of VGG16, because it retains spatial feature information.

4. **Generate the Heatmap**

   Grad-CAM uses gradients of the selected class score to identify which spatial features influenced that prediction.

5. **Compare the Results**

   We'll compare:
   - Original kidney CT image.
   - Grad-CAM heatmap.
   - Heatmap overlay on the original image.

### Grad-CAM Visualization

The following visualization illustrates how Grad-CAM helps investigate a model prediction.

```text
    Original Image
          |
          v
      Trained CNN
          |
          v
    Class Prediction
       (Cyst)
          |
          v
      Grad-CAM
          |
          v
     Heatmap
          |
          v
  Original + Heatmap
      Overlay
```

### What Does the Heatmap Tell Us?

- Red or yellow regions generally indicate stronger positive influence on the selected class score.
- Blue regions generally indicate weaker influence.
- The overlay helps us see where those influential regions are located in the original image.

For example, we can investigate whether the model focused on the suspicious kidney region or potentially misleading background features.

> [!WARNING]
> Grad-CAM shows which image regions influenced the model's prediction. It does not prove that a highlighted region contains a tumor or cyst, nor does it establish medically correct reasoning.

### Where Will We Implement Grad-CAM?

We'll create a separate Python script:

```text
research/
    |
    └── grad_cam.py
```

We'll use:

- Trained model: `artifacts/training/best_model.h5`
- Target convolutional layer: `block5_conv3`
- Technique: Grad-CAM

We won't modify or retrain our existing model.

### Our First Grad-CAM Experiment

We'll start with one misclassified image:

| Parameter | Value |
|---|---|
| Actual Class | Tumor |
| Predicted Class | Cyst |
| Prediction Confidence | 99.56% |
| Target Layer | block5_conv3 |

We'll generate its heatmap and inspect whether the model focused on relevant kidney regions or potentially misleading features.

Once we understand the visualization, we can repeat the process for other misclassified images.

### CNN vs Grad-CAM – What's the Difference?

| CNN Model | Grad-CAM |
|---|---|
| Extracts visual features | Uses convolutional features and gradients |
| Predicts a class | Highlights influential image regions |
| Produces class probabilities | Produces a heatmap |
| Answers: What does the model predict? | Helps investigate: Which regions influenced the prediction? |

**In simple terms:**

The CNN says:

> "I predict this image is a Cyst with 99.56% confidence."

Grad-CAM helps us investigate:

> "These image regions contributed most to the Cyst prediction."

Grad-CAM finds which convolutional features increased the selected class score, combines their spatial information, and produces a heatmap showing the influential regions.

> [!NOTE]
> Grad-CAM is an explainability technique, not a guarantee that the model's prediction or highlighted regions are correct.

---