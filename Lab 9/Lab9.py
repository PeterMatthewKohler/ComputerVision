# Name: Peter Kohler
# Class: CSCI 507
# Assignment: Lab 9
# Date: 10/29/21

import os
import cv2
import glob
import numpy as np

QUERY_IMAGE_DIRECTORY = 'query_images'
MINIMUM_MATCHES = 6

def main():
    # Get a list of the query images.
    assert (os.path.exists(QUERY_IMAGE_DIRECTORY))
    image_file_names = glob.glob(os.path.join(QUERY_IMAGE_DIRECTORY, "*.png"))
    assert (len(image_file_names) > 0)
    # Process each query image.
    for image_file_name in image_file_names:
        bgr_query = cv2.imread(image_file_name)
        bgr_query_gray = cv2.cvtColor(bgr_query, cv2.COLOR_BGR2GRAY)

        # Wait for xx msec (0 means wait till a keypress).
        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC


        bgr_train = cv2.imread("printer_001.png")
        bgr_train_gray = cv2.cvtColor(bgr_train, cv2.COLOR_BGR2GRAY)

        # Draw diamonds on training image to indicate points chosen
        bgr_train = cv2.drawMarker(bgr_train, (298, 245), (255, 0, 0),cv2.MARKER_DIAMOND, 20, 2)
        bgr_train = cv2.drawMarker(bgr_train, (159, 249), (255, 0, 0), cv2.MARKER_DIAMOND, 20, 2)
        bgr_train = cv2.drawMarker(bgr_train, (306, 368), (255, 0, 0), cv2.MARKER_DIAMOND, 20, 2)
        bgr_train = cv2.drawMarker(bgr_train, (461, 235), (255, 0, 0), cv2.MARKER_DIAMOND, 20, 2)
        cv2.imwrite("Annotated Training Image.jpg",bgr_train)


        # Coordinates of indicated points on training image
        Coords = np.array([ (298,245,1),
                            (159,249,1),
                            (306,368,1),
                            (461,235,1)])

        # Show input images
        cv2.imshow("Training Image", bgr_train)
        cv2.imshow("Query Image", bgr_query)


        # Initiate STAR detector
        orb = cv2.ORB_create()

        # Extract keypoints and descriptors.
        kp_train = orb.detect(bgr_train_gray)
        kp_train, desc_train = orb.compute(bgr_train_gray, kp_train)
        kp_query = orb.detect(bgr_query_gray)
        kp_query, desc_query = orb.compute(bgr_query_gray, kp_query)

        # Create brute force matcher
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)

        # Match query image descriptors to the training image.
        # Use k nearest neighbor matching and apply ratio test.
        matches = matcher.knnMatch(desc_query, desc_train, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        matches = good
        print("Number of raw matches between training and query: ", len(matches))


        # Calculate an affine transformation from the training image to the query image.
        A_train_query, inliers = calc_affine_transformation(matches, kp_train, kp_query)


        matches = [matches[i] for i in range(len(matches)) if inliers[i] == 1]

        # Apply the affine warp to warp the training image to the query image.
        if A_train_query is not None and sum(inliers) >= MINIMUM_MATCHES:
            # Preallocate numpy array
            Coords_warped = np.zeros(shape=(4,2))

            for i in range(4):
                # Perform affine transform of predetermined point
                Coords_warped[i] = A_train_query @ Coords[i].T
                # Draw diamond marker on transformed point
                bgr_query = cv2.drawMarker(bgr_query, (int(Coords_warped[i][0]), int(Coords_warped[i][1])), (255, 0, 0),cv2.MARKER_DIAMOND, 20, 2)
            cv2.imshow("Result", bgr_query)

        else:
            print("Object not detected; can't fit an affine transform")
    cv2.waitKey(0)

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
