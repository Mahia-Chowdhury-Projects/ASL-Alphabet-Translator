# ASL Alphabet Classification Pipeline

## Project Overview

This project is a computer vision pipeline that classifies ASL alphabet hand gesture images using Python, OpenCV, NumPy, and Scikit-Learn.

The pipeline loads ASL images, preprocesses them, applies data augmentation, extracts HOG features, trains a machine learning model, and evaluates the results using accuracy scores, classification reports, and confusion matrices.

## Features

- Loads ASL alphabet image datasets from folders labeled `A` through `Z`
- Converts images to grayscale
- Resizes images to `64 x 64`
- Applies CLAHE contrast enhancement
- Uses data augmentation with flips, rotations, brightness changes, and blur
- Extracts HOG descriptors for feature representation
- Supports two models:
  - Support Vector Machine
  - Random Forest
- Balances classes to reduce dataset bias
- Generates evaluation plots and confusion matrices
- Includes demo mode with synthetic data

## Dataset Format

The dataset should be organized like this:

```text
dataset/
    A/
        image1.jpg
        image2.jpg
    B/
        image1.jpg
    C/
        image1.jpg
    ...
    Z/
        image1.jpg
