# Name: Peter Kohler
# Separate Python file containing all extra functions

import cv2
import numpy as np


# Detect features in the image and return the keypoints and descriptors.
def detect_features(bgr_img, show_features=False):
    detector = cv2.xfeatures2d.SURF_create(
        hessianThreshold=100,  # default = 100
        nOctaves=4,  # default = 4
        nOctaveLayers=3,  # default = 3
        extended=False,  # default = False
        upright=False  # default = False
    )

    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))

    return keypoints, descriptors

# Calculate an affine transformation from the training image to the query image.
def calc_affine_transformation(matches_in_cluster, kp_train, kp_query):
    if len(matches_in_cluster) < 3:
        # Not enough matches to calculate affine transformation.
        return None, None
    # Estimate affine transformation from training to query image points.
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    A_train_query, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,    # Default = 3
        maxIters=2000,              # Default = 2000
        confidence=0.99,            # Default = 0.99
        refineIters=10              # Default = 10
    )
    return A_train_query, inliers


# Fuse two color images.  Assume that zero indicates an unknown value.
# At pixels where both values are known, the output is the average of the two.
# At pixels where only one is known, the output uses that value.
def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)
    # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)
    # Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0
    # Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask
    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]
    return C

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        window_name, image, point_list = param  # Unpack parameters
        cv2.rectangle(image, pt1=(x-15, y-15), pt2=(x+15, y+15), color=(0,0,255),
                      thickness=3)
        cv2.putText(image, str(len(point_list)), org=(x,y-15), color=(0,0,255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=2)
        cv2.imshow(window_name, image)
        point_list.append((x,y))

# Utility function to create an image window.
def create_named_window(window_name, image):
    # WINDOW_NORMAL allows resize; use WINDOW_AUTOSIZE for no resize.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width
    # Shrink the window if it is too big (exceeds some maximum size).
    WIN_MAX_SIZE = 8000
    if max(w, h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w, h)
    else:
        scale = 1
    cv2.resizeWindow(winname=window_name, width=int(w * scale), height=int(h * scale))

def find_corresponding(imagename1, imagename2):
    bgr_A = cv2.imread(imagename1)      # Read images
    bgr_B = cv2.imread(imagename2)
    # Create two lists.  The (x,y) points go in these lists.
    ptsA = []
    ptsB = []
    # Display images.
    displayA = bgr_A.copy()
    displayB = bgr_B.copy()
    create_named_window("Image A", displayA)
    create_named_window("Image B", displayB)
    cv2.imshow("Image A", displayA)
    cv2.imshow("Image B", displayB)
    # Assign the mouse callback function, which collects (x,y) points.
    cv2.setMouseCallback("Image A", on_mouse=get_xy, param=("Image A", displayA, ptsA))
    cv2.setMouseCallback("Image B", on_mouse=get_xy, param=("Image B", displayB, ptsB))
    # Loop until user hits the ESC key.
    print("Click on points.  Hit ESC to exit.")
    while True:
        if cv2.waitKey(100) == 27:      # ESC is ASCII code 27
            if not len(ptsA) == len(ptsB):
                print("Error: you need same number of points in both images!")
            else:
                break
    print("PtsA:", ptsA)        # Print points to the console
    print("PtsB:", ptsB)