# Chris Deal
# HW4

import sys
import cv2
#import cv2.aruco
import numpy as np


# # Your mission is to use computer vision to recognize and identify the tags,
# # and display the location of the “on/off” switch to the user. The markers are 4x4 bit patterns,
# # size is 2 inches on a side, and the markers were created by specifying 100 markers in the dictionary.
# # The marker on the left is id=0; the marker on the right is id=1.

# # Here is the location of the switch (in inches):
# # With respect to marker 0: (X,Y,Z) = (2.5; -2.0; -1.0).
# # With respect to marker 1: (X,Y,Z) = (-2.5; -2.0; -5.0).

def makePoints0(x, y, z, K, H):
    # cos >, 0, sin >
    # 0, 1, 0
    # -sin, 0, cos >  rotation
    # multipy these 2 4x4 transforms

    ax_v, ay_v, az_v = 0, np.radians(90), 0
    sx_v, sy_v, sz_v = np.sin(ax_v), np.sin(ay_v), np.sin(az_v)
    cx_v, cy_v, cz_v = np.cos(ax_v), np.cos(ay_v), np.cos(az_v)

    # set up rotation matrices
    Ry_v = np.array(((cy_v, 0, sy_v), (0, 1, 0), (-sy_v, 0, cy_v)))


    R_v =  Ry_v

    tvec = np.array([[2.5, -2.0, -1]]).T

    H2 = np.block([[R_v, tvec], [0, 0, 0, 1]])
    H_ = H @ H2
    Mext = H_[0:3, :]

    # cut off Mext row after multiplication
    P_w = np.array([x, y, z, 1])
    p = K @ Mext @ P_w
    p = p / p[2]
    p = np.round(p)  # Rounded it here

    return p


def makePoints1(x, y, z, K, H):
    # cos >, 0, sin >
    # 0, 1, 0
    # -sin, 0, cos >  rotation
    # multipy these 2 4x4 transforms

    ax_v, ay_v, az_v = 0, np.radians(-90), 0
    sx_v, sy_v, sz_v = np.sin(ax_v), np.sin(ay_v), np.sin(az_v)
    cx_v, cy_v, cz_v = np.cos(ax_v), np.cos(ay_v), np.cos(az_v)

    # set up rotation matrices

    Rx_v = np.array(((1, 0, 0), (0, cx_v, -sx_v), (0, sx_v, cx_v)))
    Ry_v = np.array(((cy_v, 0, sy_v), (0, 1, 0), (-sy_v, 0, cy_v)))
    Rz_v = np.array(((cz_v, -sz_v, 0), (sz_v, cz_v, 0), (0, 0, 1)))

    R_v = Ry_v

    tvec = np.array([[-2.5, -2.0, -5]]).T

    H2 = np.block([[R_v, tvec], [0, 0, 0, 1]])
    H_ = H @ H2
    Mext = H_[0:3, :]


    P_w = np.array([x, y, z, 1])
    p = K @ Mext @ P_w
    p = p / p[2]
    p = np.round(p)  # Rounded it here

    return p


def drawlines(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)


def main():
    currentFrame = 0

    video_capture = cv2.VideoCapture("hw4.avi")  # Open video capture object

    got_image, img = video_capture.read()

    height = img.shape[0]

    width = img.shape[1]

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    outputVideo = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=29.97,

                                  frameSize=(width, height))

    # Get the pattern dictionary for 4x4 markers, with ids 0 through 99.
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    if not got_image:
        print("Cannot read video source")

        sys.exit()

    while True:

        currentFrame += 1

        got_image, img = video_capture.read()

        if not got_image:
            break  # End of video; exit the while loop

        # 675,0,320,0,675,240,0,0,1
        K = np.array([
            (675, 0, 320),
            (0, 675, 240),
            (0, 0, 1)
        ]).astype(float)

        # Marker length
        MARKER_LENGTH = 2

        # distortion coefficient
        dist_coeff = None

        # Convert color image to gray image.
        gray_image = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)  # convert BGR to grayscale
        output_thresh, binary_image = cv2.threshold(
            src=gray_image, maxval=255,
            type=cv2.THRESH_OTSU,  # determine threshold automatically from image
            thresh=0)  # ignore this if using THRESH_OTSU

        # Observed that editing out the findCountours & drawCountours still drew the markers just fine and maybe even smoother on the
        # original image.  Also, contrary to the notes drawCountours was performed on the binary_image and this eliminated noise

        contours, hierarchy = cv2.findContours(
            image=binary_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=binary_image, contours=contours, color=(0, 0, 255), thickness=2,
                         contourIdx=-1)  # -1 means draw all contours



        # Optionally show all markers in the dictionary.
        # for id in range(0, 100):
        #     img = cv2.aruco.drawMarker(dictionary=arucoDict, id=id, sidePixels=200)

        # Detect a marker.  Returns:
        #   corners:   list of detected marker corners; for each marker, corners are clockwise)
        #   ids:   vector of ids for the detected markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=arucoDict
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                image=img, corners=corners, ids=ids, borderColor=(0, 255, 255))

            # function to compute pose from detected corners
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=MARKER_LENGTH,
                cameraMatrix=K, distCoeffs=0
            )

#        if rvecs is not None and tvecs is not None:
            # Get the pose of the first detected marker with respect to the camera.
            rvec_m_c = rvecs[0]  # This is a 1x3 rotation vector
            tm_c = tvecs[0]  # This is a 1x3 translation vector

            # funtcion to draw coordinate axes onto the image, using the detected pose
            cv2.aruco.drawAxis(
                image=img, cameraMatrix=K, distCoeffs=dist_coeff,
                rvec=rvec_m_c, tvec=tm_c, length=MARKER_LENGTH)

            # useful function to get a 3x3 rotation matrix from a rotation vector
            # You can put this into a full 4x4 transformation matrix (w/ translation) to transform the pose of the virtual object
            M_R, _ = cv2.Rodrigues(rvec_m_c)

            # translation vectors
            # tvec_0 = np.array([[2.5, -2, -1]]).T
            # tvec_1 = np.array([[-2.5, -2, -5]]).T
            # Define the homogeneous transformation matrix

            tm_c = tm_c.T

            H_pyramid = np.block([[M_R, tm_c], [0, 0, 0, 1]])  # First transformation camera to the dictionary origin

            # Apply a rotation and translation 4x4 transform

            pixel_matrix = np.zeros((5, 3))


            if currentFrame < 200:
                pixel_matrix[0] = makePoints1(-0.25, -0.25, 1.0, K, H_pyramid)
                pixel_matrix[1] = makePoints1(0.25, -0.25, 1.0, K, H_pyramid)
                pixel_matrix[2] = makePoints1(0.25, 0.25, 1.0, K, H_pyramid)
                pixel_matrix[3] = makePoints1(-0.25, 0.25, 1.0, K, H_pyramid)
                pixel_matrix[4] = makePoints1(0.0, 0.0, 0.0, K, H_pyramid)
            else:
                pixel_matrix[0] = makePoints0(-0.25, -0.25, 1.0, K, H_pyramid)
                pixel_matrix[1] = makePoints0(0.25, -0.25, 1.0, K, H_pyramid)
                pixel_matrix[2] = makePoints0(0.25, 0.25, 1.0, K, H_pyramid)
                pixel_matrix[3] = makePoints0(-0.25, 0.25, 1.0, K, H_pyramid)
                pixel_matrix[4] = makePoints0(0.0, 0.0, 0.0, K, H_pyramid)

            drawlines(img, int(pixel_matrix[0][0]), int(pixel_matrix[0][1]), int(pixel_matrix[1][0]),
                      int(pixel_matrix[1][1]))
            drawlines(img, int(pixel_matrix[1][0]), int(pixel_matrix[1][1]), int(pixel_matrix[2][0]),
                      int(pixel_matrix[2][1]))
            drawlines(img, int(pixel_matrix[2][0]), int(pixel_matrix[2][1]), int(pixel_matrix[3][0]),
                      int(pixel_matrix[3][1]))
            drawlines(img, int(pixel_matrix[3][0]), int(pixel_matrix[3][1]), int(pixel_matrix[4][0]),
                      int(pixel_matrix[4][1]))
            drawlines(img, int(pixel_matrix[4][0]), int(pixel_matrix[4][1]), int(pixel_matrix[2][0]),
                      int(pixel_matrix[2][1]))
            drawlines(img, int(pixel_matrix[4][0]), int(pixel_matrix[4][1]), int(pixel_matrix[0][0]),
                      int(pixel_matrix[0][1]))
            drawlines(img, int(pixel_matrix[4][0]), int(pixel_matrix[4][1]), int(pixel_matrix[1][0]),
                      int(pixel_matrix[1][1]))
            drawlines(img, int(pixel_matrix[4][0]), int(pixel_matrix[4][1]), int(pixel_matrix[2][0]),
                     int(pixel_matrix[2][1]))
            drawlines(img, int(pixel_matrix[3][0]), int(pixel_matrix[3][1]), int(pixel_matrix[0][0]),
                      int(pixel_matrix[0][1]))

        cv2.putText(img, str(currentFrame), org=(0, 25), fontFace=5,
                    fontScale=1.5, color=(0, 255, 255))


        #img_output = img.copy()
        #outputVideo.write(img_output)
        cv2.imshow("img", img)
        cv2.waitKey(30)

    outputVideo.release()


if __name__ == '__main__':
    main()

