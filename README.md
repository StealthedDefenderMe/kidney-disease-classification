# kidney-disease-classification deep learning project

# STEP 1: Create a template.py file first to create folder struture after cloning the repository

# STEP 2: After setting up setup.py create an virtual environment

# STEP 3: Create a virtual environment & activate. Later set setup.py file with requirement.txt

# STEP 4: Setup Logging module & exception handling (In __init__ to automatically trigger it)

# STEP 5: Import it into main.py

# STEP 6: Using python Box package we will do the exception handling inside utils folder
# inside common.py file, I'm keeping all common code i might need thrughout the application

<!-- Workflows -->
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity (Nothing but return type of any function)
5. Update the configuration manager in src config
6. Update the components (Contains data injection, model preparation & evaluation)
7. Update the pipeline (Training as well as prediction pipeline)
8. Update the main.py
9. Update the dvc.yaml (this is gonna track your entire pipeline)
10. app.py (Updating at very last)

## Here we're starting with our first component which is data injection. Here we'll inject the data

## Now the next step is to download pretrained model

# Pretrained CNN
#     ↓
# CNN feature extraction layers  ← KEEP
#     ↓
# GlobalAveragePooling2D
#     ↓
# Your Dense/ANN layers          ← ADD
#     ↓
# 4 output classes

# Pretrained CNN → the complete already-trained image model.
# Feature extraction layers → inside that model, different filters/neurons learn different visual patterns such as edges, textures, shapes, and more complex features.

<!-- 
1. Pretrained CNN

A CNN (Convolutional Neural Network) is a model designed to understand images.
Pretrained means someone has already trained it on a huge image dataset like ImageNet.
So instead of starting from zero:
Random CNN ❌
     ↓
Learn everything from scratch
we start with:
Already-trained CNN ✅
     ↓
Already knows many visual patterns

2. CNN Feature Extraction Layers
Inside the CNN are many layers that progressively learn visual features:
Early layers
→ edges, lines

Middle layers
→ shapes, textures

Deeper layers
→ complex patterns / structures

For example, the CNN might learn:
Pixels
 ↓
Edges
 ↓
Shapes
 ↓
Textures
 ↓
Complex image patterns
# We keep these layers because they are useful for understanding your kidney CT images.

Then we put our own classification layers on top to decide:
Features
   ↓
Normal / Cyst / Tumor / Stone
That's why the pretrained CNN is called the feature extractor.

3. GlobalAveragePooling2D
The CNN produces lots of feature maps. This layer summarizes each feature map into a single number.
Think:
CNN feature maps
      ↓
GlobalAveragePooling2D
      ↓
One compact list of numbers

It converts the CNN's complex visual information into something the Dense layers can easily use.

4. Your Dense / ANN layers
Now your own layers take those extracted features and learn how to classify them.
Features
   ↓
Dense layer
   ↓
Dense layer
   ↓
4 outputs

The 4 outputs represent: Normal, Cyst, Tumor, Stone
 -->


 <!-- Transformation.py 
   In transformation.py, we are basically:

   📂 Reading images from Normal, Cyst, Tumor, Stone folders.
   🔄 Resizing every image to 224 × 224.
   📦 Creating batches of 16 images.
   ✂️ Splitting the dataset into:
   80% → training
   20% → validation
   🏷️ Automatically creating labels based on the folder names.
   🎲 Using seed so the split is reproducible.

   Augmentation hasn't been added yet — we'll handle that next.
   That's it for the current transformation stage.

   # Next Stage of transformation:
      Augmentation    → Make training images varied
      VGG16 preprocess → Prepare images for VGG16
      cache()          → Avoid repeatedly loading data
      prefetch()       → Prepare next data while model trains
 -->


 <!-- What is GAP (Global Average Pooling)? -->

 <!-- VGG16 processes the image and produces 512 feature maps in your model. Each feature map represents how strongly a particular learned visual pattern appears at different spatial locations.

GAP takes each feature map and calculates its average value, producing one number per map. -->
<!-- A simple example
Imagine one feature map detects a particular edge pattern and contains these values:
\[
\begin{bmatrix}
2 & 4\\
6 & 8
\end{bmatrix}
\]

GAP calculates the average:
\[
\frac{2+4+6+8}{4}=5
\]

So, this feature map produces the number 5.

GAP repeats this process for all 512 feature maps, giving your model 512 numbers. -->

<!-- This model follows this sequence:

Kidney image → VGG16 extracts patterns → GAP produces 512 numbers → Dense(128) learns useful combinations → Dense(4, softmax) predicts the class -->

Your model has two main parts:
Layer	Purpose
VGG16 convolutional layers	Extract features from kidney images
Global Average Pooling	Summarizes feature maps into numbers
Dense(128, ReLU)	Learns combinations of those features
Dense(4, Softmax)	Predicts Cyst, Normal, Stone, or Tumor

VGG16 CNN
Extracts features from the kidney image

Global Average Pooling
Summarizes feature maps into 512 numbers

Dense(128, ReLU)
Learns useful combinations of features

Dense(4, Softmax)
Predicts Cyst, Normal, Stone, or Tumor

One important detail: VGG16 itself contains the CNN layers, while GAP and the two Dense layers form the classification head we added.

Imagine your model receives a kidney image whose correct label is Tumor.
1. Prediction: The model predicts Cyst 70%, Tumor 20%, Normal 5%, Stone 5%.
2. Calculate loss: The loss function measures how far the prediction is from the correct label, Tumor.
3. Backpropagation: The model calculates how its trainable weights contributed to the error.
4. Update weights: The optimizer (Adam in your project) adjusts those weights to reduce the loss.
5. Repeat: The model processes more images and repeats this process over multiple epochs.

In your project: You froze most VGG16 layers and unfroze the last four. The new Dense layers learn, while the unfrozen VGG16 layers can also adapt their features to your kidney images.

One key point: training aims to reduce errors, but improvement on training images alone doesn't guarantee better predictions on unseen images. That's why we monitor validation performance.

We'll use your trained best_model.h5 and the final convolutional layer of VGG16, block5_conv3, to generate the heatmap.

Important: Grad-CAM shows where the model found influential features; it does not prove that a region contains a tumor or cyst.

We'll use Grad-CAM (Gradient-weighted Class Activation Mapping) to investigate why your kidney model predicts the wrong class for some images.
How we'll do it

1. Select a misclassified image
For example, an image whose actual class is Tumor but prediction is Cyst.

2. Inspect the final convolutional layer
We'll use block5_conv3 from VGG16 because it retains spatial feature information.

3. Generate the heatmap
Grad-CAM uses gradients of the selected class score to identify image regions that influenced that prediction.

4. Compare the results

We'll compare the original image, heatmap, and overlay to see whether the model focuses on relevant regions or potentially misleading background features.

Where we'll make the changes
We'll create a separate Python script, such as research/grad_cam.py, and load your selected best_model.h5. We won't modify or retrain your model.

One important detail: Grad-CAM helps explain the model's behavior; it cannot prove that the highlighted region is medically correct.

We'll start with one Tumor → Cyst misclassified image and inspect its heatmap before analyzing more images.