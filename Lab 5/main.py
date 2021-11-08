# Name: Peter Kohler
# Assignment: Lab 5
# Course: CSCI 507
# Date: 10/1/21

import cv2
import numpy as np

# Utility function to create an image window
def create_named_window(window_name, image):
    # WINDOW_NORMAL allows resize; use WINDOW_AUTOSIZE for no resize
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width

    # Shrink the window if it is too big (exceeds some maximum size)
    WIN_MAX_SIZE = 10000
    if max(w,h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w,h)
    else:
        scale = 1
        cv2.resizeWindow(winname=window_name, width=int(w*scale), height=int(h*scale))

# Function to grab pixel coordinates from Image
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        window_name, image, point_list = param  # Unpack parameters
        cv2.rectangle(image, pt1=(x - 15, y - 15), pt2=(x + 15, y + 15), color=(0, 0, 255), thickness=3)
        cv2.putText(image, str(len(point_list)), org=(x, y - 15), color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, thickness=2)
        cv2.imshow(window_name, image)
        point_list.append((x,y))

# Function to fuse together the pixels with non black color values in two images
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
    bgr_image = cv2.imread("US_Cellular.jpg")
    create_named_window("Baseball Field", bgr_image)
    bgr_patrick = cv2.imread("patrick.webp")

    # Field height and width
    field_height = bgr_image.shape[0]
    field_width = bgr_image.shape[1]

    # Used to grab desired coordinates from image
    # cv2.setMouseCallback("Baseball Field", get_xy, param=("Baseball Field", bgr_image, Coordinates))

    # Grabbed coordinates from previously used mouseCallBack function
    Coordinates_field = np.array([(632, 387), (637, 494), (906, 480), (903, 376)])
    Coordinates_Patrick = np.array([(0, 0), (0, 471), (661, 471), (661, 0)])

    # "Remove" the sign from the image
    cv2.fillConvexPoly(bgr_image, Coordinates_field, (0, 0, 0))

    H, _ = cv2.findHomography(Coordinates_Patrick, Coordinates_field)
    warped_Patrick = cv2.warpPerspective(bgr_patrick, H, (field_width, field_height))

    # Resulting image
    patrick_and_field = fuse_color_images(warped_Patrick, bgr_image)

    cv2.imshow("Baseball Field", patrick_and_field)
    cv2.imwrite("Combined_Image.jpg",patrick_and_field)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
