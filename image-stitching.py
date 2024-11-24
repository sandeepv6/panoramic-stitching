
import PySimpleGUI as sg

#Import Libraries to be used

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    for i, img in enumerate(images):
        plt.figure()
        plt.title(f"Image {i+1}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return images