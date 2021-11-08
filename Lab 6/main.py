#Name: Peter Kohler
#Class: CSCI 507
#Assignment: Lab 6

from order_targets import order_targets
from order_targets import findClosest
import numpy as np
import cv2


def main():
    #Read in image from folder directory
    bgr_image = cv2.imread("CCCtarget.jpg")

    image_dimensions = bgr_image.shape
    image_height = image_dimensions[0]
    image_width = image_dimensions[1]


    # Create the kernel used for morphology operations
    ksize = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))

    # Font for drawing
    font = cv2.FONT_HERSHEY_SIMPLEX


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
                        # Only look for bounding boxes below a certain width threshold
                        if(w_w < 20):
                            # Displays red rectangle around black blob
                            #cv2.rectangle(img=bgr_image, pt1=(x0_w,y0_w), pt2=(x0_w+w_w, y0_w+h_w),
                            #                color=(0,0,255), thickness=2)
                            # Displays blue rectangle around white blob
                            #cv2.rectangle(img=bgr_image, pt1=(x0_b, y0_b), pt2=(x0_b + w_b, y0_b + h_b),
                            #                color=(255, 0, 0), thickness=2)

                            # Displays a '+' in the center of the white blob that resides inside of a black blob
                            cv2.putText(bgr_image, '+', (x0_w-int(0.5*w_w), y0_w+int(0.5*h_w)), font, 0.25,
                                        (0, 0, 255), 2, cv2.LINE_AA, None)
                            # Append white blob coordinates to list
                            if x0_b > 100 and y0_b > 100:
                                f_list.append(np.array([float(x0_w), float(y0_w)]))
    # Use given function to properly order the found centroids
    ordered_list = order_targets(f_list)
    o_array = np.zeros((5,2), dtype=float)
    for i in range(5):
        o_array[i][0] = ordered_list[i][0]
        o_array[i][1] = ordered_list[i][1]

    # Put display identifiers on each of the ordered points
    cv2.putText(bgr_image, '0', (int(ordered_list[0][0])-15, int(ordered_list[0][1] - 5)), font, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA, None)
    cv2.putText(bgr_image, '1', (int(ordered_list[1][0]-15), int(ordered_list[1][1] - 5)), font, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA, None)
    cv2.putText(bgr_image, '2', (int(ordered_list[2][0]-15), int(ordered_list[2][1] - 5)), font, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA, None)
    cv2.putText(bgr_image, '3', (int(ordered_list[3][0]-15), int(ordered_list[3][1] - 5)), font, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA, None)
    cv2.putText(bgr_image, '4', (int(ordered_list[4][0]-15), int(ordered_list[4][1] - 5)), font, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA, None)



    # Intrinsic Camera Characteristics
    img_width = 320
    img_height = 240
    c_x = img_width/2     #Center of image width
    c_y = img_height/2     #Center of image height
    f_x, f_y = 531, 531  #Focal length in pixels

    # Intrinsic Camera Calibration Matrix k
    K = np.array(( (f_x, 0, c_x), (0, f_y, c_y), (0, 0, 1) ))

    # Points in the model's coordinate system
    # Each column is (x, y, z, 1)
    P_M = np.array([
        [-3.7, -2.275, 0],
        [0, -2.275, 0],
        [3.7,-2.275, 0],
        [-3.7,2.275,0],
        [3.7,2.275,0]
    ])

    W = np.amax(P_M, axis=0) - np.amin(P_M, axis=0)
    L = np.linalg.norm(W)
    d = L/5

    pAxes = np.float32([
        [0, 0, 0],  # origin
        [d, 0, 0],  # x axis
        [0, d, 0],  # y axis
        [0, 0, d]  # z axis
    ])

    isPoseFound, rvec, tvec = cv2.solvePnP(P_M,o_array, K, None)
    pImg, J = cv2.projectPoints(pAxes, rvec, tvec, K, None)

    pImg = pImg.reshape(-1,2)  # reshape from size (N,1,2) to (N,2)
    cv2.line(bgr_image, tuple(np.int32(pImg[0])),tuple(np.int32(pImg[1])), (0, 0, 255), 3)  # x
    cv2.line(bgr_image, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[2])), (0, 255, 0), 3)  # y
    cv2.line(bgr_image, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[3])), (255, 0, 0), 3)  # z

    # Create new array since cv2.putText doesn't like string operations inside it's inputs
    tvecround = np.around(tvec, 1)
    cv2.putText(bgr_image, 'rvec = '+str(rvec[0])+str(rvec[1])+str(rvec[2]), (20,20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA, None)
    cv2.putText(bgr_image, 'tvec = '+str(tvecround[0])+str(tvecround[1])+str(tvecround[2]), (20, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA, None)


    cv2.imshow("Output", bgr_image)
    cv2.imwrite("Output Image.jpg", bgr_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
