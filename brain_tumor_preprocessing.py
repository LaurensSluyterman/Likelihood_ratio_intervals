import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread

# We used the BR35H data set.
def load_images_from_folder(folder, label, img_size=(128, 128, 3)):
    """This function loads and resizes the images"""
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = imread(img_path)
            img = resize(img, img_size, anti_aliasing=True, mode='reflect')
            images.append(img)
            labels.append(label)
    return images, labels

X_yes, Y_yes = load_images_from_folder('./Brain_tumor_data/yes/', 1)
X_no, Y_no = load_images_from_folder('./Brain_tumor_data/no/', 0)

all_images = np.vstack((X_yes, X_no))
all_labels = np.hstack((Y_yes + Y_no))
np.savez('./Brain_tumor_data/processed_data/data_medium', X=all_images, Y=all_labels)

# Visual check
plt.imshow(X_yes[0])
plt.show()

