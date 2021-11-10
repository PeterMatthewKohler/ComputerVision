# Name: Peter Kohler
# Class: CSCI 507
# Assignment: Homework 5
# Date: 11/9/21


import cv2
import numpy as np
from Separate_Functions import create_named_window
from Separate_Functions import detect_features
from Separate_Functions import calc_affine_transformation
from Separate_Functions import fuse_color_images


image_file_names = ["mural02.jpg", "mural03.jpg","mural04.jpg","mural05.jpg","mural06.jpg",
                    "mural07.jpg","mural08.jpg","mural09.jpg","mural10.jpg","mural11.jpg","mural12.jpg"]

my_image_names = ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg", "6.jpg"]

MINIMUM_MATCHES = 5

def main():
    # Read in image 1
    Icurrent = cv2.imread("1.jpg")

    # Get image 1 dimensions
    image1_width = Icurrent.shape[1]
    image1_height = Icurrent.shape[0]

    # Output image dimensions
    output_image_height = int(image1_height/5)
    output_image_width = image1_width

    # Points for mural orthophoto
    # pts1 = np.array([(159,126),(375,77),(298,494),(34,475)])
    # pts1_out = np.array([(0,0),(318,0),(318,435),(0,435)])
    # pts1_out[:,0] += 1*image1_width

    # Points for my photos
    pts1 = np.array([(1099, 768), (2987, 853), (2966, 2231), (1114, 2304)])
    pts1_out = np.array([(0,0),(240,0),(240,200),(0,200)])
    pts1_out[:,0]

    # Calculate homography
    H_current_mosaic, _ = cv2.findHomography(pts1, pts1_out)
    # Warp the first image
    Imosaic = cv2.warpPerspective(Icurrent, H_current_mosaic, (output_image_width,output_image_height))
    #Imosaic = cv2.warpPerspective(Icurrent, H_current_mosaic, (image1_width, image1_height))

    # Image Previous = Image Current
    Iprev = Icurrent
    # H_prev = H_current
    H_prev_mosaic = H_current_mosaic



    #---------------------------------------------------------------------------------------
    # Change image container to swap
    for image in my_image_names:
        Icurrent = cv2.imread(image)
        # Convert to grayscale
        Iprev_gray = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
        Icurrent_gray = cv2.cvtColor(Icurrent, cv2.COLOR_BGR2GRAY)



        # Initiate STAR detector
        orb = cv2.ORB_create(nfeatures=2000)

        # Extract keypoints and descriptors.
        # Prev = Train
        kp_prev = orb.detect(Iprev_gray)
        kp_prev, desc_prev = orb.compute(Iprev_gray, kp_prev)
        # Curr = Query
        kp_curr = orb.detect(Icurrent_gray)
        kp_curr, desc_curr = orb.compute(Icurrent_gray, kp_curr)

        # Create brute force matcher
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

        # Match query image descriptors to the training image.
        # Use k nearest neighbor matching and apply ratio test.
        matches = matcher.knnMatch(desc_curr, desc_prev, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        matches = good
        print("Number of raw matches between training and query: ", len(matches))

        # Assign Source and Destination Keypoints
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in matches])#.reshape(-1,1,2)
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in matches])#.reshape(-1,1,2)

        # Calculate Homography
        H_current_prev, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        # Calculate the updated Mosaic Homography
        H_current_mosaic = H_prev_mosaic @ H_current_prev
        # Warp the current image
        Icurrent_warp = cv2.warpPerspective(Icurrent, H_current_mosaic,(output_image_width,output_image_height))
        # Fuse the mosaic with the warped current image creating the new mosaic image
        Imosaic = fuse_color_images(Imosaic, Icurrent_warp)
        create_named_window("Mosaic",Imosaic)
        cv2.imshow("Mosaic", Imosaic)
        cv2.waitKey(0)
        # Update the new image
        Iprev = Icurrent
        # Update the new mosaic homography
        H_prev_mosaic = H_current_mosaic

    # Create final mosaic photo
    cv2.imwrite("MY PHOTO Mosaic Output.jpg", Imosaic)


if __name__ == '__main__':
    main()
