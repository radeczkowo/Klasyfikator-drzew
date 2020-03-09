import os.path
import cv2
import random
from scipy import ndimage
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


gen = ImageDataGenerator(rotation_range=5, width_shift_range=(-0.1, 0.1), shear_range=0.15,
                         brightness_range=(1.0, 1.15), vertical_flip=False,
                         zoom_range=(0.85,  1), channel_shift_range=10, horizontal_flip=True, fill_mode='nearest',
                         height_shift_range=(-0.1, 0.1))
"""

gen = ImageDataGenerator(rotation_range=5, width_shift_range=(-0.1, 0.1), shear_range=0.5,
                         brightness_range=(1.0, 1.15), vertical_flip=False,
                         zoom_range=(0.85,  1), channel_shift_range=10, horizontal_flip=True, fill_mode='nearest',
                         height_shift_range=(-0.1, 0.1))

"""
def rozszerzaniedanych(img, tdata, number, IMG_SIZE, rrange):
    H = img.shape[0]
    W = img.shape[1]
    img = np.reshape(img, (-1, H, W, 1))
    aug_iter = gen.flow(img)
    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(rrange)]
    for i in range(rrange):
        img = cv2.resize(aug_images[i], (IMG_SIZE, IMG_SIZE))
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.imshow(img, cmap="gray")
        #plt.show()
        #print number
        tdata.append([img, number])
        print len(tdata)


