# Name: Peter Kohler
# Class: CSCI 507
# Assignment: HW3

from order_targets import order_targets
import numpy as np
import cv2

def fuse_color_images(A,B):
    assert (A.ndim == 3 and B.ndim == 3)
    assert (A.shape == B.shape)

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

def main():
    # Read in image to perform homography transform on
    bgr_patrick = cv2.imread("patrick.webp")

    # Coordinates for homography on Patrick Image
    patrick_Width = bgr_patrick.shape[0]
    patrick_Height = bgr_patrick.shape[1]

    # Coordinates_Patrick = np.array([(0, 0), (661, 0), (661, 471), (0, 471)])
    Coordinates_Patrick = np.array([(0, 0), (patrick_Height, 0), (patrick_Height, patrick_Width), (0, patrick_Width)])

    # Read in images from a video file in current folder
    video_capture = cv2.VideoCapture("fiveCCC.mp4") # Open video capture object
    got_image, bgr_image = video_capture.read() # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        sys.exit()
    video_dimensions = bgr_image.shape
    video_height = video_dimensions[0]
    video_width = video_dimensions[1]
    # Read and show images until end of video is reached.

    # Video creation declaration
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Video Output.avi", fourcc=fourcc, fps=30.0, frameSize=(video_width, video_height))

    # Create the kernel used for morphology operations
    ksize = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Font for drawing
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Frame counter
    frame = 0

    while True:
        # Increment Frame Count
        frame += 1

        # Read in frame from video file
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop

        # Convert image to grayscale
        gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

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

        # Find connected components and region properties (white and black blobs)
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
                if np.sqrt((abs(x0_w - x0_b)) ** 2 + (abs(y0_w - y0_b)) ** 2) <= 10 and (w_b * h_b) > 20:
                    # Check if top left corner of white bounding box is inside black bounding box
                    if (x0_b < x0_w) and (y0_b < y0_w):
                        # Check if bottom right corner of white bounding box is inside black bounding box
                        if (x0_w + w_w) < (x0_b + w_b) and (y0_w + h_w) < (y0_b + h_b):
                            # Append white blob coordinates to list
                            f_list.append(np.array([float(x0_w), float(y0_w)]))

        if len(f_list) == 5:
            # Use provided function to properly order the found centroids in specific order
            # Order is:     0   1   2
            #               3       4
            ordered_list = order_targets(f_list)
            o_array = np.zeros((5, 2), dtype=float)
            for i in range(5):
                o_array[i][0] = ordered_list[i][0]
                o_array[i][1] = ordered_list[i][1]

            # Remove Portion of the target that will be replaced by new image
            # o_fillpoly array goes from top left, to top right, to bot right, to bot left
            o_fillpoly = np.array([ordered_list[0], ordered_list[2], ordered_list[4], ordered_list[3]], dtype=np.int32)
            cv2.fillConvexPoly(bgr_image, o_fillpoly,(0,0,0))

            # Perform homography transform
            H , _ = cv2.findHomography(Coordinates_Patrick, o_fillpoly)
            warped_Patrick = cv2.warpPerspective(bgr_patrick, H, (video_width, video_height))

            patrick_and_video = fuse_color_images(warped_Patrick, bgr_image)

            # Intrinsic Camera Characteristics
            img_width = 320
            img_height = 240
            c_x = img_width / 2  # Center of image width
            c_y = img_height / 2  # Center of image height
            f_x, f_y = 531, 531  # Focal length in pixels

            # Intrinsic Camera Calibration Matrix k
            K = np.array(((f_x, 0, c_x), (0, f_y, c_y), (0, 0, 1)))

            # Points in the model's coordinate system
            P_M = np.array([
                [-3.7, -2.275, 0],
                [0, -2.275, 0],
                [3.7, -2.275, 0],
                [-3.7, 2.275, 0],
                [3.7, 2.275, 0]
            ])

            # Scaling the length of the origin axis vectors
            W = np.amax(P_M, axis=0) - np.amin(P_M, axis=0)
            L = np.linalg.norm(W)
            d = L / 3

            pAxes = np.float32([
                [0, 0, 0],  # origin
                [d, 0, 0],  # x axis
                [0, d, 0],  # y axis
                [0, 0, d]  # z axis
            ])

            # OpenCv function to find the pose
            isPoseFound, rvec, tvec = cv2.solvePnP(P_M, o_array, K, None)
            # OpenCv function to project 3D image points onto a plane
            pImg, J = cv2.projectPoints(pAxes, rvec, tvec, K, None)

            pImg = pImg.reshape(-1, 2)  # reshape from size (N,1,2) to (N,2)
            cv2.line(patrick_and_video, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[1])), (0, 0, 255), 3)  # x axis
            cv2.line(patrick_and_video, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[2])), (0, 255, 0), 3)  # y axis
            cv2.line(patrick_and_video, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[3])), (255, 0, 0), 3)  # z axis

            # Display the rotation vector
            cv2.putText(patrick_and_video, 'rvec = ' + str(rvec[0]) + str(rvec[1]) + str(rvec[2]), (20, video_height-50),
                        font, 0.5, (0, 255, 255), 1, cv2.LINE_AA, None)

            # Create new array since cv2.putText doesn't like string operations inside it's inputs
            # This is done to cut off all but 1 decimal place for translation vector
            tvecrounded = np.around(tvec, 1)

            # Display the translation vector
            cv2.putText(patrick_and_video,
                        'tvec = ' + str(tvecrounded[0]) + str(tvecrounded[1]) + str(tvecrounded[2]), (20, video_height-20),
                        font, 0.5, (0, 255, 255), 1, cv2.LINE_AA, None)

            # Display the frame number
            cv2.putText(patrick_and_video,
                        str(frame), (20, 50),
                        font, 1, (0, 255, 255), 2, cv2.LINE_AA, None)
            #Show the image
            cv2.imshow("Output", patrick_and_video)
            videoWriter.write(patrick_and_video)
            cv2.waitKey(5)

    videoWriter.release()

if __name__ == '__main__':
    main()


