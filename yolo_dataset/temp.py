import os
import random
import shutil

def split_data(img_folder, label_folder, train_folder, val_folder, val_size=0.2):
    """
    Randomly splits images and their corresponding label files into training and validation sets,
    and saves them into separate 'images' and 'labels' subfolders.

    Parameters:
        img_folder (str): Folder containing the images (.jpg files).
        label_folder (str): Folder containing the labels (.txt files).
        train_folder (str): Folder to save training images and labels.
        val_folder (str): Folder to save validation images and labels.
        val_size (float): Proportion of the data to be used for validation (default is 20%).
    """
    # Create directories for training and validation sets, including subfolders for images and labels
    os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_folder, "labels"), exist_ok=True)

    # Get lists of image and label files
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')]
    label_files = [f for f in os.listdir(label_folder) if f.lower().endswith('.txt')]

    # Ensure the number of images and labels match
    assert len(image_files) == len(label_files), "Number of images and labels do not match!"

    # Shuffle the list of image files for randomness
    random.shuffle(image_files)

    # Split into training and validation sets
    num_val = int(len(image_files) * val_size)
    val_files = image_files[:num_val]
    train_files = image_files[num_val:]

    # Move files to their respective folders
    for file in val_files:
        # Move image to validation images folder
        img_src = os.path.join(img_folder, file)
        img_dst = os.path.join(val_folder, "images", file)
        shutil.copy(img_src, img_dst)

        # Move corresponding label to validation labels folder
        label_file = os.path.splitext(file)[0] + ".txt"
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(val_folder, "labels", label_file)
        shutil.copy(label_src, label_dst)

    for file in train_files:
        # Move image to training images folder
        img_src = os.path.join(img_folder, file)
        img_dst = os.path.join(train_folder, "images", file)
        shutil.copy(img_src, img_dst)

        # Move corresponding label to training labels folder
        label_file = os.path.splitext(file)[0] + ".txt"
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(train_folder, "labels", label_file)
        shutil.copy(label_src, label_dst)

    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")

# Example usage
img_folder = "temp/images"  # Folder containing images
label_folder = "temp/labels"  # Folder containing labels
train_folder = "train"  # Folder to save training images and labels
val_folder = "val"  # Folder to save validation images and labels

split_data(img_folder, label_folder, train_folder, val_folder, val_size=0.2)
