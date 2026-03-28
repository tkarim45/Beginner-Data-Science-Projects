# Eye Disease Detection

Classifies eye disease images into five categories (Glaucoma, Cataracts, Uveitis, Crossed Eyes, Bulging Eyes) using transfer learning with a fine-tuned ResNet34 model. Includes a separate image augmentation pipeline that generates flipped and rotated copies of the original dataset to increase training data.

## Dataset

Eye disease images organized into five class folders. The original dataset is augmented via horizontal flips and 180-degree rotations (see `augment_images.ipynb`). Both the Original Dataset and Augmented Dataset are included.

## Tech Stack

- fastai / fastcore
- PyTorch (via fastai)
- Pillow
- pandas
- numpy
- matplotlib

## Results

A fine-tuned ResNet34 model is trained for 50 epochs on the augmented dataset to classify five eye disease categories: Glaucoma, Cataracts, Uveitis, Crossed Eyes, and Bulging Eyes. See notebook for detailed results.

## How to Run

1. Install dependencies: `pip install fastai fastcore pandas numpy`
2. Place your dataset in the `Augmented Dataset/` folder with one subfolder per class.
3. Run `augment_images.ipynb` first if you need to generate augmented images from the `Original Dataset/`.
4. Open `model.ipynb`, update the `path` variable to point to your dataset, and run all cells.
