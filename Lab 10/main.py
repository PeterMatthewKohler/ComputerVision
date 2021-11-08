# Name: Peter Kohler
# Class: CSCI 507
# Assignment: Lab 10
# Date: 11/5/21

import cv2
import numpy as np
import math
from vanishing import find_vanishing_pt
from vanishing import find_vanishing_point_directions

def main():
    for image in ["chess.jpg", "corridor1.jpg", "corridor2.jpg", "corridor3.png"]:

        MAX_WIDTH = 1000  # Shrink image to this width, if image is wider (don't need full size)


        bgr_img = cv2.imread(image)
        if bgr_img.shape[1] > MAX_WIDTH:
            s = MAX_WIDTH / bgr_img.shape[1]
            bgr_img = cv2.resize(bgr_img, dsize=None, fx=s, fy=s)
        image_width = bgr_img.shape[1]
        image_height = bgr_img.shape[0]

        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Smooth the image with a Gaussian filter.  If sigma is not provided, it
        # computes it automatically using   sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
        gray_img = cv2.GaussianBlur(
            src=gray_img,
            ksize=(1, 1),  # kernel size (should be odd numbers; if 0, compute it from sigma)
            sigmaX=0, sigmaY=0)

        # Pick a threshold such that we get a relatively small number of edge points.
        thresh_canny = 200
        MIN_FRACT_EDGES = 0.05
        MAX_FRACT_EDGES = 0.08

        edge_img = cv2.Canny(
            image=gray_img,
            apertureSize=3,  # size of Sobel operator
            threshold1=thresh_canny,  # lower threshold
            threshold2=3 * thresh_canny,  # upper threshold
            L2gradient=True)  # use more accurate L2 norm
        # cv2.imshow("Hello", edge_img)
        cv2.imwrite(str(image+" Edge Image.jpg"), edge_img)

        while np.sum(edge_img) / 255 < MIN_FRACT_EDGES * (image_width * image_height):
            print("Decreasing threshold ...")
            thresh_canny *= 0.9
            edge_img = cv2.Canny(
                image=gray_img,
                apertureSize=3,  # size of Sobel operator
                threshold1=thresh_canny,  # lower threshold
                threshold2=3 * thresh_canny,  # upper threshold
                L2gradient=True)  # use more accurate L2 norm
        while np.sum(edge_img) / 255 > MAX_FRACT_EDGES * (image_width * image_height):
            print("Increasing threshold ...")
            thresh_canny *= 1.1
            edge_img = cv2.Canny(
                image=gray_img,
                apertureSize=3,  # size of Sobel operator
                threshold1=thresh_canny,  # lower threshold
                threshold2=3 * thresh_canny,  # upper threshold
                L2gradient=True)  # use more accurate L2 norm



        # Run Hough transform.  The output hough_lines has size (N,1,2), where N is #lines.
        # The 3rd dimension has values rho,theta for the line.
        MIN_HOUGH_VOTES_FRACTION = 0.04  # Minimum points on a line (as fraction of image width)
        MIN_LINE_LENGTH_FRACTION = 0.06

        # Define a threshold for the peak finder.
        houghThresh = int(image_width * MIN_HOUGH_VOTES_FRACTION)
        # Run Hough transform.  The output houghLines has size (N,1,4), where N is #lines.
        # The 3rd dimension has the line segment endpoints: x0,y0,x1,y1.
        houghLines = cv2.HoughLinesP(
            image=edge_img,
            rho=1,
            theta=math.pi / 180,
            threshold=houghThresh,
            lines=None,
            minLineLength=int(image_width * MIN_LINE_LENGTH_FRACTION),
            maxLineGap=10)
        print("Found %d line segments" % len(houghLines))

        # For visualizing the lines, draw on a grayscale version of the image.
        bgr_display = cv2.cvtColor(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        for i in range(0, len(houghLines)):
            l = houghLines[i][0]
            cv2.line(bgr_display, (l[0], l[1]), (l[2], l[3]), (0, 0, 255),
                     thickness=2, lineType=cv2.LINE_AA)
        find_vanishing_point_directions(houghLines,bgr_display, 3, None)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
