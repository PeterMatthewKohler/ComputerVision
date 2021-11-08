# Name: Peter Kohler
# Class: CSCI 507
# Assignment: Homework 4
# Date: 10/19/21

import cv2
import numpy as np
import sys

# Function call to create Rotation Matrix
def rotation(X_rot, Y_rot, Z_rot):  # Rotation inputs in radians
    # Rotation Inputs
    ax, ay, az = X_rot, Y_rot, Z_rot  # radians

    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

    if X_rot != 0 and Y_rot == 0 and Z_rot == 0:
        R = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))  # Rotation about x axis
    elif X_rot == 0 and Y_rot != 0 and Z_rot == 0:
        R = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))  # Rotation about y axis
    elif X_rot == 0 and Y_rot == 0 and Z_rot != 0:
        R = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))  # Rotation about z axis
    elif X_rot == 0 and Y_rot == 0 and Z_rot == 0:
        R = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))      # Returns identity matrix if rotation = 0
    return R


def main():
    # Read in images from video file
    video_capture = cv2.VideoCapture("hw4.avi") # Open video capture object
    got_image, bgr_image = video_capture.read() # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        sys.exit()
    image_dimensions = bgr_image.shape
    image_height = image_dimensions[0]
    image_width = image_dimensions[1]


    # Video creation declaration
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("HW4 Video Output.avi", fourcc = fourcc, fps = 30.0,
                                        frameSize=(image_width, image_height))

    # Get the pattern dictionary for 4x4 markers, with ids 0 through 99
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    # Intrinsic Camera Characteristics
    c_x = image_width/2     # Center of image width
    c_y = image_height/2    # Center of image height
    f_x, f_y = 675, 675     # Focal lengths in pixels

    # Intrinsic Camera Calibration Matrix k
    K = np.array(( (f_x, 0, c_x), (0, f_y, c_y), (0, 0, 1) ))

    # MARKER_LENGTH Definition (The height and width of dictionary image in real world)
    MARKER_LENGTH = 2.0 # inches

    # Pyramid Points to create drawing with
    Coords = np.array(([-0.25, -0.25, 1.0, 1.0], [0.25, -0.25, 1.0, 1.0], [0.25, 0.25, 1.0, 1.0], [-0.25, 0.25, 1.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0]))

    # Copy of Pyramid points to iterate with through frames in video
    Coords_proj = np.zeros(Coords.shape, dtype=float)

    # Create rotation matrix going from Dictionary ID 1 to switch
    # Rotation about y axis -pi/2 rad
    rot_1 = rotation(0, -np.pi / 2, 0)
    # Create rotation matrix going from Dictionary ID 0 to switch
    # Rotation about y axis pi/2 rad
    rot_0 = rotation(0, np.pi / 2, 0)

    while True:
        # Read and show images until end of video is reached.
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break # End of video; exit the while loop

        # Detect a marker. Returns:
        #   corners:   list of detected marker corners; for each marker, corners are clockwise)
        #   ids:   vector of ids for the detected markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=bgr_image,
            dictionary=arucoDict
        )

        if ids is not None:
            # Draw on detected markers
            cv2.aruco.drawDetectedMarkers(
                image=bgr_image, corners=corners, ids=ids, borderColor=(0, 0, 255))

            # Draw pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=MARKER_LENGTH,
                cameraMatrix=K, distCoeffs=None
            )
            # Get the pose of the first detected marker with respect to the camera.
            rvec_m_c = rvecs[0]  # This is a 1x3 rotation vector
            tm_c = tvecs[0].T  # This is a 1x3 translation vector
            # Draw Origin Axis' (Red is X axis, Green is Y axis, Blue is Z axis)
            cv2.aruco.drawAxis(
                image=bgr_image, cameraMatrix=K, distCoeffs=None,
                rvec=rvec_m_c, tvec=tm_c, length=MARKER_LENGTH)

            # Generate 3x3 Rotation Matrix from Rotation Vector
            rot, _ = cv2.Rodrigues(rvec_m_c)

            # Generate full 4x4 Transformation matrix with respect to Dictionary Image
            Transform = np.block([[rot, tm_c], [0, 0, 0, 1]])

            if ids[0][0] == 1:  # ID = 1
                # Transformation from Dictionary Image ID 1 location to Button
                # Translation to Switch with respect to marker 1: (X,Y,Z) = (-2.5; -2.0; -5.0)
                trans_1 = np.array([[-2.5],
                                   [-2.0],
                                   [-5.0]])

                # 4x4 Transform from Dictionary ID 1 to Switch
                Transform_1 = np.block([[rot_1, trans_1],[0, 0, 0, 1]])
                # Calculate final Transform
                Transform = Transform @ Transform_1

            elif ids[0][0] == 0:   # ID = 0
                # Transformation from Image ID 0 location to Button
                # Translation to Switch with respect to marker 0: (X,Y,Z) = (2.5; -2.0; -1.0)
                trans_0 = np.array([[2.5],
                                   [-2.0],
                                   [-1.0]])
                # 4x4 Transform from Dictionary ID 0 to Switch
                Transform_0 = np.block([[rot_0, trans_0],[0, 0, 0, 1]])
                # Calculate final Transform
                Transform = Transform @ Transform_0

            # Camera External Transformation Matrix
            M_ext = Transform[0:3, :]

            # Project Pyramid Coordinates onto Image Coordinates
            for i in range(5):
                Coords_proj[i][0:3] = K @ M_ext @ Coords[i].T
                Coords_proj[i][0:3] = Coords_proj[i][0:3] / Coords_proj[i][2]
                for j in range(2):
                    Coords_proj[i][j] = round(Coords_proj[i][j], 2)
            # Draw the arch
            cv2.line(bgr_image, (int(Coords_proj[0][0]), int(Coords_proj[0][1])), (int(Coords_proj[4][0]), int(Coords_proj[4][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[1][0]), int(Coords_proj[1][1])), (int(Coords_proj[4][0]), int(Coords_proj[4][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[2][0]), int(Coords_proj[2][1])), (int(Coords_proj[4][0]), int(Coords_proj[4][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[3][0]), int(Coords_proj[3][1])), (int(Coords_proj[4][0]), int(Coords_proj[4][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[4][0]), int(Coords_proj[4][1])), (int(Coords_proj[0][0]), int(Coords_proj[0][1])),
                     (0, 0, 255), 2)
            # Draw the base
            cv2.line(bgr_image, (int(Coords_proj[0][0]), int(Coords_proj[0][1])), (int(Coords_proj[1][0]), int(Coords_proj[1][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[1][0]), int(Coords_proj[1][1])), (int(Coords_proj[2][0]), int(Coords_proj[2][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[2][0]), int(Coords_proj[2][1])), (int(Coords_proj[3][0]), int(Coords_proj[3][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[3][0]), int(Coords_proj[3][1])), (int(Coords_proj[4][0]), int(Coords_proj[4][1])),
                     (0, 0, 255), 2)
            cv2.line(bgr_image, (int(Coords_proj[0][0]), int(Coords_proj[0][1])), (int(Coords_proj[3][0]), int(Coords_proj[3][1])),
                     (0, 0, 255), 2)


        cv2.imshow("Output", bgr_image)
        videoWriter.write(bgr_image)
        cv2.waitKey(50)
    videoWriter.release()

if __name__ == '__main__':
    main()

