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


def detect_and_match_features(img1, img2, use_sift=True, use_flann=False):
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

    # Use FLANN or BF Matcher
    if use_flann:
        if use_sift:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict()
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) if use_sift else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img)
    plt.title("Feature Matching")
    plt.show()

    return kp1, kp2, matches

def compute_homography_and_warp(kp1, kp2, matches, img1, img2):
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute homography using RANSAC
    H, inliers = compute_homography_ransac(src_pts, dst_pts)

    # Warp the image
    height, width = img2.shape[:2]
    warped_img = warp_image(img1, H, (height, width))

    # Display the warped image
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image")
    plt.show()

    return warped_img

def compute_homography(src_pts, dst_pts):
    num_points = src_pts.shape[0]
    A = []

    for i in range(num_points):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]

        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]  # Normalize

    return H

def compute_homography_ransac(src_pts, dst_pts, max_iters=2000, threshold=5.0):
    max_inliers = []
    final_H = None

    for _ in range(max_iters):
        idx = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        H = compute_homography(src_sample, dst_sample)

        src_pts_hom = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
        projected_pts = (H @ src_pts_hom.T).T
        projected_pts = projected_pts / projected_pts[:, [2]]

        errors = np.linalg.norm(dst_pts - projected_pts[:, :2], axis=1)
        inliers = errors < threshold

        if np.sum(inliers) > np.sum(max_inliers):
            max_inliers = inliers
            final_H = H

        if np.sum(max_inliers) > 0.8 * len(src_pts):
            break

    if final_H is None:
        raise ValueError("Could not compute homography matrix.")

    return final_H, max_inliers

def warp_image(img, H, output_shape):
    H_inv = np.linalg.inv(H)
    height, width = output_shape
    warped_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x_coords)
    coords = np.stack([x_coords, y_coords, ones], axis=-1).reshape(-1, 3)

    # Apply inverse homography
    src_coords = H_inv @ coords.T
    src_coords = src_coords / src_coords[2, :]
    src_coords = src_coords[:2, :].T

    # Interpolate pixel values
    src_x = src_coords[:, 0]
    src_y = src_coords[:, 1]

    valid_idx = (src_x >= 0) & (src_x < img.shape[1]) & (src_y >= 0) & (src_y < img.shape[0])
    valid_coords = coords[valid_idx]
    src_x_valid = src_x[valid_idx]
    src_y_valid = src_y[valid_idx]

    # Bilinear interpolation
    src_x0 = np.floor(src_x_valid).astype(np.int32)
    src_x1 = src_x0 + 1
    src_y0 = np.floor(src_y_valid).astype(np.int32)
    src_y1 = src_y0 + 1

    # Clip coordinates to image dimensions
    src_x0 = np.clip(src_x0, 0, img.shape[1]-1)
    src_x1 = np.clip(src_x1, 0, img.shape[1]-1)
    src_y0 = np.clip(src_y0, 0, img.shape[0]-1)
    src_y1 = np.clip(src_y1, 0, img.shape[0]-1)

    Ia = img[src_y0, src_x0]
    Ib = img[src_y1, src_x0]
    Ic = img[src_y0, src_x1]
    Id = img[src_y1, src_x1]

    wa = (src_x1 - src_x_valid) * (src_y1 - src_y_valid)
    wb = (src_x1 - src_x_valid) * (src_y_valid - src_y0)
    wc = (src_x_valid - src_x0) * (src_y1 - src_y_valid)
    wd = (src_x_valid - src_x0) * (src_y_valid - src_y0)

    warped_pixels = (Ia.T * wa + Ib.T * wb + Ic.T * wc + Id.T * wd).T

    # Assign pixels to warped image
    warped_img.reshape(-1, 3)[valid_idx] = warped_pixels

    return warped_img

def blend_images(img1, img2):
    blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
    plt.title("Blended Image")
    plt.show()
    return blended_img

def stitch_images(images, use_sift=True, use_flann=False):
    base_img = images[0]
    for i in range(1, len(images)):
        kp1, kp2, matches = detect_and_match_features(base_img, images[i],use_sift=use_sift, use_flann=use_flann)
        warped_img = compute_homography_and_warp(kp1, kp2, matches, base_img, images[i])
        base_img = blend_images(warped_img, images[i])
    return base_img


def panoramic_gui():
    stitched_image = None  # Variable to store the stitched image

    layout = [
        [sg.Text("Select Images for Panorama")],
        [sg.Input(key="FILES", enable_events=True, visible=False), sg.FilesBrowse("Browse", file_types=(("Image Files", "*.jpg;*.png"),))],
        [sg.Text("Choose Feature Detector:"), sg.Combo(["SIFT", "ORB"], default_value="SIFT", key="DETECTOR")],
        [sg.Text("Choose Matcher:"), sg.Combo(["BF Matcher", "FLANN"], default_value="BF Matcher", key="MATCHER")],
        [sg.Button("Stitch Images"), sg.Button("Save Image", disabled=True), sg.Exit()],
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
                detector = values["DETECTOR"]
                matcher = values["MATCHER"]

                images = load_images(image_paths)
                stitched_image = stitch_images(images, use_sift=(detector == "SIFT"), use_flann=(matcher == "FLANN"))
                plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
                plt.title("Panoramic Image")
                plt.show()
                window["STATUS"].update("Stitching completed!")
                window["Save Image"].update(disabled=False)  # Enable Save Image button
            except Exception as e:
                window["STATUS"].update(f"Error: {e}")
        elif event == "Save Image":
            if stitched_image is not None:
                save_path = sg.popup_get_file("Save Image As", save_as=True, file_types=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png")))
                if save_path:
                    try:
                        save_image(stitched_image, save_path)
                        sg.popup("Image saved successfully!")
                    except Exception as e:
                        sg.popup(f"Error saving image: {e}")
                else:
                    sg.popup("Save operation canceled.")
            else:
                sg.popup("No image to save. Please stitch images first.")
    
    window.close()
def save_image(image, output_path="panorama_output.jpg"):
    cv2.imwrite(output_path, image)
    print(f"Panorama saved at {output_path}")