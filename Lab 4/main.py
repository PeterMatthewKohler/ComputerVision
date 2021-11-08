#Name: Peter Kohler
#Course: CSCI 507
#Assignment: Lab 4
#Date: 9/17/21

import numpy as np
import cv2


def lab_func(input_img, low_thresholds, high_thresholds):
    #Read input_img from directory and create HSV conversions

    bgr_image = cv2.imread(input_img)
    image_height = bgr_image.shape[0]
    image_width = bgr_image.shape[1]

    #convert to HSV image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    #split into different bands
    planes = cv2.split(hsv_image)

    #Used for trackbar creation (not needed in final image creation
    #windowNames = ["Hue Image", "Saturation Image", "Gray Image"]
    #for i in range(3):
    #    cv2.namedWindow(windowNames[i])

    #Create trackbars
    #for i in range(3):
    #    cv2.createTrackbar("Low", windowNames[i], low_thresholds[i], 255, nothing)
    #    cv2.createTrackbar("High", windowNames[i], high_thresholds[i], 255, nothing)

    while True:
        #create output thresholded image
        thresh_img = np.full((image_height, image_width), 255, dtype=np.uint8)
        for i in range(3):
        #    low_val = cv2.getTrackbarPos("Low", windowNames[i])
        #    high_val = cv2.getTrackbarPos("High", windowNames[i])

        #    _,low_img = cv2.threshold(planes[i], low_val, 255, cv2.THRESH_BINARY)
        #    _,high_img = cv2.threshold(planes[i], high_val, 255, cv2.THRESH_BINARY_INV)

            _, low_img = cv2.threshold(planes[i], low_thresholds[i], 255, cv2.THRESH_BINARY)
            _, high_img = cv2.threshold(planes[i], high_thresholds[i], 255, cv2.THRESH_BINARY_INV)

            thresh_band_img = cv2.bitwise_and(low_img, high_img)
        #    cv2.imshow(windowNames[i], thresh_band_img)

            # AND with output thresholded image
            thresh_img = cv2.bitwise_and(thresh_img, thresh_band_img)

        #cv2.imshow("Output Thresholded Image", thresh_img)

        #create Disk structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
        #Opening Morphological Operation
        filtered_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        #Closing Morphological Operation
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)

        cv2.imshow("Cleaned Up Image", filtered_img)
        cv2.imwrite(input_img+" Filtered", filtered_img)
        if not cv2.waitKey(100) == -1:
            break

def nothing(x):
    pass

if __name__ == '__main__':
    low_threshold = [136, 50, 52]
    high_threshold = [255, 255, 255]
    img_names = ["stop0.jpg", "stop1.jpg", "stop2.jpg", "stop3.jpg", "stop4.jpg"]

    print("Threshold Values Used: \t\t\t(H, S, V)")
    print("Low: \t\t\t\t\t\t", low_threshold)
    print("High: \t\t\t\t\t\t", high_threshold)

    print("Size of Morphological Elements Used: 5x10 Ellipse")


    for i in range(5):
        lab_func(img_names[i], low_threshold, high_threshold)


