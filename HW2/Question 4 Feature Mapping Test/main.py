# Name: Peter Kohler
# Course: CSCI 507
# Assignment: HW 2 Question 4
import cv2
import numpy as np
import sys
from operator import itemgetter

def main():
    # Read in images from a video file in current folder
    video_capture = cv2.VideoCapture("fiveCCC.mp4") # Open video capture object
    got_image, bgr_image = video_capture.read() # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        sys.exit()
    image_dimensions = bgr_image.shape
    image_height = image_dimensions[0]
    image_width = image_dimensions[1]
    # Read and show images until end of video is reached.

    # Video creation declaration
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Video Output.avi", fourcc = fourcc, fps = 30.0,
                                        frameSize=(image_width, image_height))

    # Create the kernel used for morphology operations
    ksize = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))

    # Font for drawing
    font = cv2.FONT_HERSHEY_SIMPLEX

    # List to contain initial centroids used for feature mapping extra credit
    start_list=[]
    # Frame count
    frame = 0

    while True:
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop
        # Convert image to grayscale
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        """#Do Otsu's thresholding
        thresh, binary_img = cv2.threshold(gray_img, thresh=0, maxval=255,
                                           type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)"""

        # Do adaptive thresholding
        binary_img = cv2.adaptiveThreshold(src=gray_img,
                                           maxValue=255,
                                           adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                           thresholdType=cv2.THRESH_BINARY,
                                           blockSize=51,
                                           C=-10)

        # Clean up using Opening + Closing
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # Find connected components and region properties (white blobs)
        num_labels_w, labels_img_w, stats_w, centroids_w = cv2.connectedComponentsWithStats(binary_img)
        num_labels_b, labels_img_b, stats_b, centroids_b = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary_img))

        # Feature mapping coord list
        f_list = []

        # Iterate through white blobs finding centroids
        for stat_w, centroid_w in zip(stats_w, centroids_w):
            x0_w = stat_w[cv2.CC_STAT_LEFT]
            y0_w = stat_w[cv2.CC_STAT_TOP]
            w_w = stat_w[cv2.CC_STAT_WIDTH]
            h_w = stat_w[cv2.CC_STAT_HEIGHT]

            # Iterate through black blobs looking for centroids close together
            for stat_b, centroid_b in zip(stats_b, centroids_b):

                x0_b = stat_b[cv2.CC_STAT_LEFT]
                y0_b = stat_b[cv2.CC_STAT_TOP]
                w_b = stat_b[cv2.CC_STAT_WIDTH]
                h_b = stat_b[cv2.CC_STAT_HEIGHT]

                # White is inside box
                # Black is outside box

                # Check distance between white centroid and black centroid
                if np.sqrt((abs(x0_w - x0_b))**2 + (abs(y0_w - y0_b))**2) <= 10 and (w_b*h_b) > 20:
                    # Check if top left corner of white bounding box is inside black bounding box
                    if (x0_b < x0_w) and (y0_b < y0_w) :
                        # Check if bottom right corner of white bounding box is inside black bounding box
                        if (x0_w + w_w) < (x0_b + w_b) and (y0_w + h_w) < (y0_b + h_b):

                            # Displays red rectangle around black blob
                            #cv2.rectangle(img=bgr_image_display, pt1=(x0_w,y0_w), pt2=(x0_w+w_w, y0_w+h_w),
                            #                color=(0,0,255), thickness=2)
                            # Displays blue rectangle around white blob
                            #cv2.rectangle(img=bgr_image_display, pt1=(x0_b, y0_b), pt2=(x0_b + w_b, y0_b + h_b),
                            #                color=(255, 0, 0), thickness=2)

                            # Displays a '+' in the center of the white blob that resides inside of a black blob
                            cv2.putText(bgr_image, '+', (x0_w-int(0.5*w_w), y0_w+int(0.5*h_w)), font, 0.25,
                                        (0, 0, 255), 2, cv2.LINE_AA, None)
                            # Append white blob coordinates to list
                            if x0_b > 100 and y0_b > 100:
                                f_list.append([x0_w, y0_w])

        if frame < 1:
            top_row = []
            bot_row = []
            for i in f_list:
                if i[1] > 145 and i[1] < 151:
                    top_row.append(i)
                else:
                    bot_row.append(i)
            # Sort based on which is further to the left
            top_sorted = sorted(top_row, key=itemgetter(0))
            bot_sorted = sorted(bot_row, key=itemgetter(0))
            start_list = [top_sorted[0],top_sorted[1],top_sorted[2],bot_sorted[0],bot_sorted[1]]

        if len(f_list)>=5:
            for i in range(5):  # iterate through start_list
                for j in range(5):  # iterate through f_list
                    if (np.sqrt((abs(start_list[i][0] - f_list[j][0]))**2 + (abs(start_list[i][1] - f_list[j][1]))**2)) < 14:
                        start_list[i] = f_list[j]

        # Display identifiers
        cv2.putText(bgr_image, '0', (start_list[0][0], start_list[0][1]-5), font, 0.75,
                    (0,0,255), 2, cv2.LINE_AA, None)
        cv2.putText(bgr_image, '1', (start_list[1][0], start_list[1][1]-5), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA, None)
        cv2.putText(bgr_image, '2', (start_list[2][0], start_list[2][1]-5), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA, None)
        cv2.putText(bgr_image, '3', (start_list[3][0], start_list[3][1]-5), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA, None)
        cv2.putText(bgr_image, '4', (start_list[4][0], start_list[4][1]-5), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA, None)

        # Display Frame Number
        cv2.putText(bgr_image, str(frame), (20, 40), font, 1,
                    (0, 255, 0), 2, cv2.LINE_AA, None)
        # Show the current frame
        #cv2.imshow("Output", bgr_image)
        # Write the frame to an output video file
        videoWriter.write(bgr_image)
        cv2.waitKey(50)

        frame+=1
    videoWriter.release()


if __name__ == '__main__':
    main()

