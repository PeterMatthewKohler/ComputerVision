#Lab 1 for CSCI 507
#By Peter Kohler

import cv2
import sys

def main():
    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture("earth.wmv")     # Open video capture object
    got_image, bgr_image = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()
    image_dimensions = bgr_image.shape
    image_height = image_dimensions[0]
    image_width = image_dimensions[1]
    # Read and show images until end of video is reached.

    c_y = 270
    c_x = 480
    Z = 1
    frameCount = 1

    #video creation declaration
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter("output_Peter_Kohler.avi", fourcc=fourcc, fps = 30.0, frameSize =(image_width, image_height))
    while True:
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break       # End of video; exit the while loop

        #Adding 4 red dots
        cv2.putText(bgr_image, str(frameCount), (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.circle(bgr_image, (int(((-1/Z)*500)+c_x), int(((-1/Z)*500)+c_y)), 5, (0, 0, 255), -1)
        cv2.circle(bgr_image, (int(((1 / Z) * 500) + c_x), int(((-1 / Z) * 500) + c_y)), 5, (0, 0, 255), -1)
        cv2.circle(bgr_image, (int(((1 / Z) * 500) + c_x), int(((1 / Z) * 500) + c_y)), 5, (0, 0, 255), -1)
        cv2.circle(bgr_image, (int(((-1 / Z) * 500) + c_x), int(((1 / Z) * 500) + c_y)), 5, (0, 0, 255), -1)
        cv2.imshow("Video Output", bgr_image)
        Z += 0.1
        frameCount+=1

        videoWriter.write(bgr_image)


        # Wait for xx msec (0 means wait till a keypress).
        cv2.waitKey(30)
    videoWriter.release()

if __name__ == "__main__":
    main()
