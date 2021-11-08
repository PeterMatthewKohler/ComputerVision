#CSCI 507 Lab 2
#By: Peter Kohler


import numpy as np
import cv2


def main():

    #Section 1 ----------------------------------
    ax, ay, az = 1.1, -0.5, 0.1 #radians

    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

    Rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))   #Rotation about x axis
    Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))   #Rotation about y axis
    Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))   #Rotation about z axis

    # Apply X rotation first, then Y, then Z
    R = Rz @ Ry @ Rx    # Use @ for matrix mult


    #Part a.)
    print("Rotation Matrix R\n", R)

    #Part b.)
    print("\nR^T\n", R.T)
    print("\nR^-1\n", np.linalg.inv(R))

    #Part c.)
    print("\n ZYX Rotation Matrix:\n",Rx @ Ry @ Rz)

    #Section 2 -----------------------------------
    #Part a.)
    #Translation from World to Camera
    t_Wo_C = np.array([[10, -25, 40]]).T
    #Rotation Matrix
    R_Wo_C = Rz @ Ry @ Rx
    #Homogenous Transformation that represents pose of Camera w/ respect to World
    H_Wo_C = np.block([[R_Wo_C, t_Wo_C], [0, 0, 0, 1]])
    print("\nTransform from World to Camera:\n",H_Wo_C)

    #Part b.)
    H_C_Wo = np.linalg.inv(H_Wo_C)
    print("\nTransform from Camera to World\n", H_C_Wo)

    #Part c.)
    c_x = 256/2     #Center of image width
    c_y = 170/2     #Center of image height
    f_x, f_y = 400, 400  #Focal length in pixels

    #Intrinsic Camera Calibration Matrix k
    k = np.array(( (f_x, 0, c_x), (0, f_y, c_y), (0, 0, 1) ))
    print("\nIntrinsic Camera Calibration Matrix k\n", k)


    #Part d.) Blank Zeroes image
    image = np.zeros((170, 256, 3), np.uint8)


    #Extrinsic Camera Matrix
    M_ext = H_C_Wo[0:3, :]

    #Points in World Frame
    P1 = np.array([6.8158, -35.1954, 43.0640, 1]).T
    P2 = np.array((7.8493, -36.1723, 43.7815, 1)).T
    P3 = np.array((9.9579, -25.2799, 40.1151, 1)).T
    P4 = np.array((8.8219, -38.3767, 46.6153, 1)).T
    P5 = np.array((9.5890, -28.8402, 42.2858, 1)).T
    P6 = np.array((10.8082, -48.8146, 56.1475, 1)).T
    P7 = np.array((13.2690, -58.0988, 59.1422, 1)).T

    #Project 7 Points to Image
    Proj_1 = k @ M_ext @ P1
    Proj_1 = Proj_1 / Proj_1[2]

    Proj_2 = k @ M_ext @ P2
    Proj_2 = Proj_2 / Proj_2[2]

    Proj_3 = k @ M_ext @ P3
    Proj_3 = Proj_3 / Proj_3[2]

    Proj_4 = k @ M_ext @ P4
    Proj_4 = Proj_4 / Proj_4[2]

    Proj_5 = k @ M_ext @ P5
    Proj_5 = Proj_5 / Proj_5[2]

    Proj_6 = k @ M_ext @ P6
    Proj_6 = Proj_6 / Proj_6[2]

    Proj_7 = k @ M_ext @ P7
    Proj_7 = Proj_7 / Proj_7[2]


    #Section 3 -----------------------------------------------------------------
    cv2.line(image, (int(Proj_1[0]), int(Proj_1[1])), (int(Proj_2[0]), int(Proj_2[1]) ), (255, 255, 255), 2 )
    cv2.line(image, (int(Proj_2[0]), int(Proj_2[1])), (int(Proj_3[0]), int(Proj_3[1])), (255, 255, 255), 2)
    cv2.line(image, (int(Proj_3[0]), int(Proj_3[1])), (int(Proj_4[0]), int(Proj_4[1])), (255, 255, 255), 2)
    cv2.line(image, (int(Proj_4[0]), int(Proj_4[1])), (int(Proj_5[0]), int(Proj_5[1])), (255, 255, 255), 2)
    cv2.line(image, (int(Proj_5[0]), int(Proj_5[1])), (int(Proj_6[0]), int(Proj_6[1])), (255, 255, 255), 2)
    cv2.line(image, (int(Proj_6[0]), int(Proj_6[1])), (int(Proj_7[0]), int(Proj_7[1])), (255, 255, 255), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
