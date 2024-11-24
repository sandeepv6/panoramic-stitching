#Import Libraries to be used

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg

def load_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    for i, img in enumerate(images):
        plt.figure()
        plt.title(f"Image {i+1}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return images



def panoramic_gui():
    layout = [
        [sg.Text("Select Images for Panorama")],
        [sg.Input(key="FILES", enable_events=True, visible=False), sg.FilesBrowse("Browse", file_types=(("Image Files", "*.jpg;*.png"),))],
        [sg.Button("Stitch Images"), sg.Exit()]
    ]

    window = sg.Window("Panoramic Stitching Tool", layout)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Stitch Images":
            image_paths = values["FILES"].split(";")
            images = load_images(image_paths)
            stitched_image = stitch_images(images)
            plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
            plt.title("Panoramic Image")
            plt.show()
    
    window.close()
