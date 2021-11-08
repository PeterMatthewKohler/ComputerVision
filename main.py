import cv2
import urllib.request

def main():
    # Download an image from the web and save it to a file.
    url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2019/12/03202400/Yellow-Labrador-Retriever.jpg"
    urllib.request.urlretrieve(url, "myimage.jpg")

    img = cv2.imread("myimage.jpg")
    cv2.imshow("My image", img)
    cv2.waitKey(0)

    print("All done!")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()