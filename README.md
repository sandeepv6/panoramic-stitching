# **Panoramic Image Stitching Tool**

## **Overview**

The Panoramic Image Stitching Tool is a Python-based application that combines overlapping images into seamless panoramic views. This project leverages computer vision algorithms for feature detection, homography estimation, image warping, and blending, all accessible through an intuitive graphical interface.

## **Features**

- **Feature Detection Options**: Choose between SIFT and ORB for feature detection.
- **Matching Algorithms**: Supports BF Matcher and FLANN Matcher for feature correspondence.
- **Manual Feature Selection**: Manually select points for stitching to handle complex cases.
- **User-Friendly Interface**: PySimpleGUI-based GUI for easy image input, algorithm selection, and image saving.
- **Image Blending**: Smooth transitions between stitched images for a visually appealing panorama.

## **Technologies Used**

- **Python**
- **OpenCV**: Computer vision library for feature detection, homography, warping, and blending.
- **NumPy**: Efficient numerical operations.
- **PySimpleGUI**: For building the graphical user interface.
- **Matplotlib**: For visualizing results.

## **Getting Started**

### **Prerequisites**
- Python 3.7 or later
- Required libraries:
  ```bash
  pip install opencv-python-headless numpy matplotlib PySimpleGUI
