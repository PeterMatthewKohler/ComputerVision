#Name: Peter Kohler
#Course: CSCI 507
#Assignment: HW2 Question 2

import cv2
import numpy as np

def main():
    #Read in image from directory
    bgr_image = cv2.imread("textsample.tif")
    img_height, img_width = bgr_image.shape[:2]

    #User defined first instance of letter 'a' used as a template to find other instances of letter a
    template_coord = [683, 141]

    #Template used in Cross-Correlation
    template_image = bgr_image[ template_coord[1]-10 : template_coord[1]+10, template_coord[0]-7 : template_coord[0]+8 ]
    template_h, template_w = template_image.shape[:2]
    cv2.imwrite("Q2Template.jpg", template_image)
    #Perform normalized cross correlation with template image
    C = cv2.matchTemplate(bgr_image, template_image, cv2.TM_CCOEFF_NORMED)

    #Filter out all coordinates of matches below threshold value
    matches = np.where(C >= 0.7)

    #Create a mask to help filter out letters we have already detected
    mask = np.zeros((img_height, img_width, 3), np.uint8)

    #Create a counter for detected letters
    count = 0

    #Iterates through each coordinate in list of matches and draws a rectangle around it
    for i in range(len(matches[1])):
        # Looks for value 255 in color value at pixel coordinate location in mask image
        if 255 not in mask[matches[0][i], matches[1][i]]:
            #Paints over the coordinate location with a shape of the template and slightly larger in the color white in the mask image
            cv2.rectangle(mask, (matches[1][i]-2, matches[0][i]-2), (matches[1][i] + template_w, matches[0][i] + template_h), (255, 255, 255),-1)
            #Creates rectangle around the detected letter a
            cv2.rectangle(bgr_image, (matches[1][i], matches[0][i]), (matches[1][i] + template_w, matches[0][i] + template_h), (0, 0, 255),2)
            #Adds to letter a count
            count += 1

    cv2.imshow("Image", bgr_image)
    cv2.imwrite("Q2Output.jpg", bgr_image)

    print("There are ", count, " instances of the letter 'a' in this image.")
    cv2.waitKey(0)

if __name__ == '__main__':
    main()


