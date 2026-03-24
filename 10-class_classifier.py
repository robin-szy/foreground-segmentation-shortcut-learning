"""
10-Class Classifier
Subject: Introduction to Deep Learning
Author: Robin Szymanski
Dataset: https://www.kaggle.com/datasets/aryanpandey1109/inaturalist12k

About the dataset:
The dataset is from Kaggle (originally iNaturalist) and contains 12000 images in total.
In the training set, each of the following 10 categories has 1000 images: Amphibia, Arachnida,
Fungi, Mammalia, Plantae, Animalia, Aves,  Insecta, Mollusca, Reptilia. The validation set has
the same categories, but 200 images per category.
"""

"""
Todos:
- HPC: Check DL test scripts -> Talk with HPC team on Tuesday
- Build a first small model based on lecture
- Research on how other people have done it
- num_workers: Measure how many are good. Currently 4. Maybe it's too much, maybe too less?
"""

"""
My current idea for the project is:

1) Create a model on my own (no transfer learning) and train it on the images

2) The most successful model from step 1 will be retrained on a set of images which I feature engineered as follows: 
The background of the image might be telling the model too much about the class (e.g. a blue sky as a background will probably tell the model that something is a bird). I want to segment the objects of the images, and then train the model only on the cut out images. Then I want to compare the model trained on feature-engineered data with the first model. In particular, the training procedure should look like:
- 1 time cropped training, cropped test set
- 1x cropped training, non-cropped test set
- 1x non-cropped training, cropped test set
- 1x non-cropped training, non-cropped test set

3) The outcome will likely be that model 1 is better. So I want to create some artificial images with the segmented objects, but using backgrounds (e.g. a cow with a sky background). Maybe, I can show by this that for such specific edge cases, model 2 performs better.

4) Optional: Use the feature-engineered data and change the background, a little bit as in number 3. So, I augment my data by taking backgrounds from other images.

5) Optional: Use an existing model (transfer learning) and compare results.
"""

"""
Architecture:
Resize: Important for fully connected layer: transforms.Resize((128, 128))
Augment data: Rotation, zoom (at beginning of each epoch).
Normalization of input data!

# =====================
# Feature Extractor
# =====================
- Conv: nn.Conv2d(3, 16, 3)
(- BatchNorm): Aurelien Geron: p.367ff, maybe use TFLite's converter -> Slower training, faster conversion.
    But it's optional. Could also worsen it (ML professor)
- ReLu
- Conv
(- BatchNorm)
- ReLu
- Pooling
- Conv
(- BatchNorm)
- ReLu
- Conv
- ReLu
- Pooling
- MLP
    - Fourier feature embedding
    - Sliding window weights update (stabilizing)
    - Temporal segmentation?
    

# =====================
# Classifier
# =====================
- Flatten
- Fully connected layers
- Softmax

# =====================
# Training
# =====================
- Dropout (maybe not with BatchNorm together, it depends, see Geron Aurelien. 
  Prob better to use BatchNorm, then dropout if overfitting)
  But machine learning professor said use dropout for sure, because images can have mistakes.
  Dropout not in every layer
  Rather something like this:
  [Conv -> BN -> ReLU] x N
    -> (maybe Dropout here if needed)
    -> Fully Connected -> ReLU -> Dropout
    -> Output
- Gradient Clipping
- Early Stopping
- Skip Connections for sure!!!

Metrics: AUC, loss, Cross Entropy


Maybe retrain model with 3 different splits
"""

# ====================
# Libraries
# ====================

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ===================
# Parameters
# ===================
DIR_SPLITS = "splits/split_seed_42/"
BATCH_SIZE = 32
EPOCHS = 10


# ====================
# Loading the dataset
# ====================

"""
PyTorch's Dataset and DataLoader:
Dataset stores the samples and their corresponding labels, and DataLoader wraps an 
iterable around the Dataset to enable easy access to the samples.
"""



# I prefer to have full control here, as each index is assigned one of the folders
# By automatically creating the indices, I'm scared that the folders might be
# differently arranged
CLASSES = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia"
]
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}


train_transform = transforms.Compose([
    transforms.Resize(256),  # shortest side = 256
    transforms.CenterCrop(224),  # 224x224
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),  # shortest side = 256
    transforms.CenterCrop(224),  # 224x224
    transforms.ToTensor(),
])


# More details, see https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# It needs to have this structure, because DataLoader only accepts two different types of
# datasets: map-style and iterable-style. Below is a map-style one (__len__ and __getitem__
# necessary). More: https://docs.pytorch.org/docs/stable/data.html
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir=".", transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # With this, you can access data by e.g. train_dataset[0]
        row = self.df.iloc[idx]

        img_path = self.root_dir / row["filepath"]
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        label = class_to_idx[row["label"]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
        # DON'T DO THIS: return image.to("cuda"), label -> If this is together with num_workers -> Bad idea
        # "it is probably not a good idea to call .cuda() in the Dataset object, as it will have to move each
        # sample (rather than the batch) to GPU separately, incurring a lot of overhead."
        # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work


train_dataset = CustomDataset(DIR_SPLITS + "train.csv", transform=train_transform)
val_dataset   = CustomDataset(DIR_SPLITS + "val.csv", transform=val_test_transform)
test_dataset  = CustomDataset(DIR_SPLITS + "test.csv", transform=val_test_transform)


# Troubleshooting
# Memory overload? Maybe it's due to setting num_workers, see here:
# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
# Also pin_memory can lead to errors if not enough RAM,
# see: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Multi-process data loading (how many subprocesses)
    pin_memory=True # PyTorch usually copies data from CPU memory to GPU (slower). With pin_memory
                    # data is allocated in pinned (page-locked) RAM, GPU can transfer data more
                    # efficiently via Direct Memory Access. Only do if enough RAM.
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)