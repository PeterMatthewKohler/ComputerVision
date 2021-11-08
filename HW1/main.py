#Name: Peter Kohler
#Course: CSCI 507
#Assignment: Homework 1
import numpy as np
import matplotlib.pyplot as plt
import cv2



#Function call to create Rotation Matrix
def rotation(X_rot, Y_rot, Z_rot):  #rotation inputs in radians
    #Rotation Inputs
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
        R = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))      #Returns identity matrix if rotation = 0
    return R

#Function call to create Translation Matrix
def translation(x_dist, y_dist, z_dist):
    return np.array([[x_dist, y_dist, z_dist]]).T

#Function call to create Homogenous Transformation
def Homog_T(Rot_param, Transl_param):
    return np.block([[Rot_param, Transl_param], [0, 0, 0, 1]])



if __name__ == '__main__':
    #Problem 3 ---------------------------------

    #Transformation from World to Vehicle
    H_W_V = Homog_T(rotation(0, 0, (np.pi/6)) , translation(6, -8, 1))

    #Transformation from Vehicle to Mount
    H_V_M = Homog_T(rotation((-2*np.pi/3), 0, 0), translation(0, 0, 3))

    #Transformation from Mount to Camera
    H_M_C = Homog_T(rotation(0, 0, 0), translation(0, -1.5, 0))

    #Transformation from World to Camera
    H_W_C = H_W_V @ H_V_M @ H_M_C

    #Transformation from Camera to World
    H_C_W = np.linalg.inv(H_W_C)

    print(H_W_V)
    print(H_V_M)
    print(H_M_C)
    print(H_C_W)

    #Pyramid Points in World Frame
    Coords = np.array(([-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 3.0, 1.0]))  #Zero based indices

    #Intrinsic Camera Characteristics
    img_width = 640
    img_height = 480
    c_x = img_width/2     #Center of image width
    c_y = img_height/2     #Center of image height
    f_x, f_y = 600, 600  #Focal length in pixels

    #Intrinsic Camera Calibration Matrix k
    k = np.array(( (f_x, 0, c_x), (0, f_y, c_y), (0, 0, 1) ))

    #Extrinsic Camera Matrix
    M_ext = H_C_W[0:3, :]

    #Project Pyramid Coordinates onto Image Coordinates
    for i in range(5):
        Coords[i][0:3] = k @ M_ext @ Coords[i].T

        Coords[i][0:3] = Coords[i][0:3]/Coords[i][2]
        for j in range(2):
            Coords[i][j] = round(Coords[i][j])

    #The image coordinates of the 5 projected points
    print(Coords)
    #Create empty white image
    img = np.zeros((img_height, img_width, 3), np.uint8)  #height x width

    #Creating the arch
    cv2.line(img, (int(Coords[0][0]), int(Coords[0][1])), (int(Coords[4][0]), int(Coords[4][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[1][0]), int(Coords[1][1])), (int(Coords[4][0]), int(Coords[4][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[2][0]), int(Coords[2][1])), (int(Coords[4][0]), int(Coords[4][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[3][0]), int(Coords[3][1])), (int(Coords[4][0]), int(Coords[4][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[4][0]), int(Coords[4][1])), (int(Coords[0][0]), int(Coords[0][1])), (255, 255, 255), 2 )
    #Creating the base
    cv2.line(img, (int(Coords[0][0]), int(Coords[0][1])), (int(Coords[1][0]), int(Coords[1][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[1][0]), int(Coords[1][1])), (int(Coords[2][0]), int(Coords[2][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[2][0]), int(Coords[2][1])), (int(Coords[3][0]), int(Coords[3][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[3][0]), int(Coords[3][1])), (int(Coords[4][0]), int(Coords[4][1])), (255, 255, 255), 2 )
    cv2.line(img, (int(Coords[0][0]), int(Coords[0][1])), (int(Coords[3][0]), int(Coords[3][1])), (255, 255, 255), 2 )

    cv2.imshow("Pyramid", img)
    cv2.imwrite("Pyramid.jpg",img)
    cv2.waitKey(0)

    #3D coordinates lists for pose plotting
    x_coord = []
    y_coord = []
    z_coord = []
    origin_pose_x = []
    origin_pose_y = []
    origin_pose_z = []

    #Homogenous Transformation Matrix for World Origin(Just Identity Matrix)
    World_Origin = np.array(([1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))
    #Transformation from World to Mount
    H_W_M = H_W_V@H_V_M

    test = [World_Origin, H_W_V, H_W_M, H_W_C]

    test_size = len(test)
    for i in test:
        x_coord.append(i[0][3])
        y_coord.append(i[1][3])
        z_coord.append(i[2][3])
        for x in range(3):
            origin_pose_x.append(i[x][0])
        for y in range(3):
            origin_pose_y.append(i[y][1])
        for z in range(3):
            origin_pose_z.append(i[z][2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set(xlim = (-2,8), ylim = (-8,2), zlim = (0, 6))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    for i in [0,3,6,9]:

        #Creates Origin Vector in X direction
        ax.quiver(x_coord[int(i/3)], y_coord[int(i/3)], z_coord[int(i/3)], origin_pose_x[i], origin_pose_x[i+1], origin_pose_x[i+2], length=1)
        #Creates Origin Vector in Y direction
        ax.quiver(x_coord[int(i/3)], y_coord[int(i/3)], z_coord[int(i/3)], origin_pose_y[i], origin_pose_y[i+1], origin_pose_y[i+2], length=1)
        #Creates Origin Vector in Z direction
        ax.quiver(x_coord[int(i/3)], y_coord[int(i/3)], z_coord[int(i/3)], origin_pose_z[i], origin_pose_z[i+1], origin_pose_z[i+2], length=1)

    ax.text(World_Origin[0][3],World_Origin[1][3],World_Origin[2][3]-0.5, "Pyramid/World Origin")
    ax.text(H_W_V[0][3], H_W_V[1][3], H_W_V[2][3] - 0.5, "Vehicle Origin")
    ax.text(H_W_M[0][3], H_W_M[1][3], H_W_M[2][3] - 0.5, "Mount Origin")
    ax.text(H_W_C[0][3], H_W_C[1][3], H_W_C[2][3] + 0.5, "Camera Origin")
    plt.show()
    fig.savefig('scatterplot.png')

