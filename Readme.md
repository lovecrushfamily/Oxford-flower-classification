

# Oxford Flower 102 



#

## Motivation

This project will put an end to my endless learning curve these days, I'll leverage all of those techniques, knowledge I've learned so far to put on this project for real, 

I'm not interest on car, plant, animal, transport,..such those things out there. Flowers are all I needed, So I came up with idea using the biggest flower dataset I got on Kaggle, (Actually I found out this dataset first then the Idea come later to me).
By the way, let's get this shit done.


- This project was supposed to run on Kaggle, 'Cause my local machine was just supported with Nvidia 1060, so I mean it's so weak I can feel, But don't worry, I also take advantage of it, find out the limitation of my GPU.

- There's alot of thing concerned me with Kaggle, I'm not gonna run this project in cloud anymore, Use cloud computing for good.


#### oke my self, just hear me out clearly,

- First 1, running on local was a bad move => Running all of code on Kaggle for good, and convenient

- Commit on local, but not push on github

- Using kaggle as the main Jupiter server 

- Why : 

    high computational power

    big input and outout source of storage, up to 20GB, too much, must utilize it.

    Everything run into a notebook, I like it, Maybe we have to choose going modular method for good, 

    Kaggle's not really have good syncing process with Github, just commit the notebook directly not pull the new code from local.

    So use Kaggle from now on for good, do not touching into the notebook_dl in local for good..

- Kaggle: notebook deep learning, Commit only from Kaggle, not from local or Colab, I don't want to make any wrong move anymore.

- colab : notebook machine learning

- Local : doing something else.


### *Update*

- Change to the modular to keep the kaggle notebook more clean and clarity to us, 

- And mimic the Pytoch Zero to Hero style, I think It has became standard for decade., or you can make your own way to implement it.

- Make sure to code anything day by day, this head can't remember anything if missed two days. Never miss two days in a rows.







| Step | What to Do |
|---------------------|-----------------------------------------------------------------|
| Data | Use Oxford Flowers dataset, original splits if possible |
| Preprocessing | Resize, normalize images |
| Feature Extraction | SIFT, HOG, Color Histograms, BoVW |
| Feature Encoding | Cluster descriptors, build histograms (if using SIFT/HOG) |
| Classification | SVM (linear/RBF), Random Forest, etc. |
| Evaluation | Accuracy, mean per-class accuracy, use original splits |


To replicate the work from the original Oxford Flowers dataset paper (especially if focusing on traditional machine learning, not deep learning), you should follow a pipeline similar to what the authors did. Here’s a step-by-step guide tailored for traditional ML, based on the typical methodology in the Oxford Flowers papers:

1. Understand the Dataset

Oxford Flowers 17/102: Contains images of flowers, each labeled with a class (species).
Data splits: The original paper uses specific train/val/test splits. Try to use the same splits if possible (they are often provided with the dataset).

2. Image Preprocessing

Resize images to a standard size (e.g., 128x128 or 224x224).
Color normalization (optional, but can help).
Augmentation is less common in traditional ML, but you can try simple flips/rotations.

3. Feature Extraction

Traditional ML does not use raw pixels. You need to extract features from images. The original paper and many classical approaches use:
Color histograms (e.g., RGB, HSV histograms)
SIFT (Scale-Invariant Feature Transform) descriptors
HOG (Histogram of Oriented Gradients)
Bag of Visual Words (BoVW): Cluster SIFT/HOG descriptors and represent images as histograms of visual words.
Other descriptors: Gabor filters, LBP (Local Binary Patterns), etc.

4. Feature Encoding (if needed)

If using SIFT/HOG, you often need to cluster descriptors (e.g., k-means) to create a codebook (visual vocabulary).
Represent each image as a histogram over this codebook (BoVW).

5. Classification

Train a classifier on the extracted features:
SVM (Support Vector Machine) is the most common and was used in the original paper.
Other options: Random Forest, k-NN, Logistic Regression.
Tune hyperparameters (e.g., SVM kernel, C, gamma).

6. Evaluation

Use the same metrics as the paper (usually accuracy, sometimes mean per-class accuracy).
Use the same data splits for fair comparison.

7. Comparison & Analysis

Compare your results to those reported in the paper.
Analyze which features and classifiers work best.