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