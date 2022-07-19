import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle


# from csv read in the training images and steering labels
def read_csv(path):
    data_csv = pd.read_csv(path)
    images_path_center = np.array(data_csv['center'])
    labels_center = np.array(data_csv['steering'])
    images_path_left = np.array(data_csv['left'])
    labels_left = labels_center+0.25
    images_path_right = np.array(data_csv['right'])
    labels_right = labels_center - 0.25
    images_path = np.concatenate((images_path_center, images_path_right, images_path_left))
    labels = np.concatenate((labels_center, labels_right, labels_left))
    return shuffle(images_path, labels)


# create an image generator
def generator(data, label, batch_size=64):
    data_size = len(data)
    while True:
        data, label = shuffle(data, label)
        for start in range(0, data_size, batch_size):
            data_path_batch = data[start:start + batch_size]
            label_batch = label[start:start + batch_size]
            data_batch = []
            for index, path in enumerate(data_path_batch):
                image = cv2.imread(path.strip())
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                float_image = image.astype('float32')
                data_batch.append(float_image)
            data_batch = np.array(data_batch)
            yield shuffle(data_batch, label_batch)

