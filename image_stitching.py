#Import Libraries to be used

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import argparse

def load_images(image_paths):
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image at {path} could not be loaded.")
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    if not images:
        raise ValueError("No valid images loaded. Please check the paths.")
    return images


def detect_and_match_features(img1, img2, use_sift=True):
    if use_sift:
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    else:
        # Initialize ORB detector
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Use FLANN or Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) if use_sift else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img)
    plt.title("Feature Matching")
    plt.show()
    
    return kp1, kp2, matches

def compute_homography_and_warp(kp1, kp2, matches, img1, img2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width, channels = img2.shape
    warped_img = cv2.warpPerspective(img1, H, (width, height))
    
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image")
    plt.show()
    
    return warped_img

def blend_images(img1, img2):
    blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
    plt.title("Blended Image")
    plt.show()
    return blended_img

def stitch_images(images):
    base_img = images[0]
    for i in range(1, len(images)):
        kp1, kp2, matches = detect_and_match_features(base_img, images[i])
        warped_img = compute_homography_and_warp(kp1, kp2, matches, base_img, images[i])
        base_img = blend_images(warped_img, images[i])
    return base_img


def panoramic_gui():
    layout = [
        [sg.Text("Select Images for Panorama")],
        [sg.Input(key="FILES", enable_events=True, visible=False), sg.FilesBrowse("Browse", file_types=(("Image Files", "*.jpg;*.png"),))],
        [sg.Button("Stitch Images"), sg.Exit()],
        [sg.Text("", size=(40, 1), key="STATUS")]
    ]

    window = sg.Window("Panoramic Stitching Tool", layout)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Stitch Images":
            window["STATUS"].update("Processing...")
            try:
                image_paths = values["FILES"].split(";")
                images = load_images(image_paths)
                stitched_image = stitch_images(images)
                plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
                plt.title("Panoramic Image")
                plt.show()
                window["STATUS"].update("Stitching completed!")
            except Exception as e:
                window["STATUS"].update(f"Error: {e}")
    
    window.close()
    
def save_image(image, output_path="panorama_output.jpg"):
    cv2.imwrite(output_path, image)
    print(f"Panorama saved at {output_path}")


