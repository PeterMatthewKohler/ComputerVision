#Course: CSCI 507
#Name: Peter Kohler
#Assignment: Lab 3

import numpy as np
import cv2
import sys

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param[0] = x
        param[1] = y

def main():
    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture("building.avi")  # Open video capture object
    got_image, bgr_image = video_capture.read()  # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        sys.exit()
    image_dimensions = bgr_image.shape
    image_height = image_dimensions[0]
    image_width = image_dimensions[1]
    # Read and show images until end of video is reached.

    # video creation declaration
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Video Output.avi", fourcc=fourcc, fps=30.0,
                                  frameSize=(image_width, image_height))

    #Empty list for Mouse callback function
    Coordinate = [0,0]
    frame = 1

    #Show image and select coordinate from mouse click
    cv2.imshow("Video Output", bgr_image)
    cv2.setMouseCallback("Video Output", get_xy, Coordinate)

    #Pause 2 seconds
    if frame == 1:
        cv2.waitKey(2000)

    # Create Template
    template_image = bgr_image[Coordinate[1]-25 : Coordinate[1]+25, Coordinate[0]-25 : Coordinate[0]+25]
    cv2.rectangle(bgr_image, (Coordinate[0] - 25, Coordinate[1] - 25), (Coordinate[0] + 25, Coordinate[1] + 25),(0, 0, 255), 2)

    while True:

        got_image, bgr_image = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop

        C = cv2.matchTemplate(bgr_image, template_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
        (start_x, start_y) = max_loc
        cv2.rectangle(bgr_image, (start_x, start_y), (start_x+template_image.shape[1], start_y+template_image.shape[0]),
                      (0, 0, 255), 2)       #Start of template image is in top left corner, not where you click!

        cv2.imshow("Video Output", bgr_image)
        videoWriter.write(bgr_image)
        cv2.waitKey(60)    # Wait for xx msec (0 means wait till a keypress).
        frame+=1
    videoWriter.release()

if __name__ == '__main__':
    main()


