import cv2
import numpy as np

img1 = cv2.imread("../ScreenShots/image2.jpg")
img2 = cv2.imread("../ScreenShots/image2.jpg")

# Resize images to the same height (optional)
height = max(img1.shape[0], img2.shape[0])
width1 = int(img1.shape[1] * (height / img1.shape[0]))
width2 = int(img2.shape[1] * (height / img2.shape[0]))
image1 = cv2.resize(img1, (width1, height))
image2 = cv2.resize(img2, (width2, height))

mask = np.zeros_like(img1)  # Initialize mask

# Variables
ix = -1
iy = -1
drawing = False


def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw_rectangle(ix, iy, x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False


def draw_rectangle(x1, y1, x2, y2):
    cv2.rectangle(mask, pt1=(x1, y1),
                  pt2=(x2, y2),
                  color=(255, 255, 255),  # White color for mask
                  thickness=-1)


def main():
    cv2.namedWindow(winname="Title of Popup Window")
    cv2.setMouseCallback("Title of Popup Window", draw_mask)

    while True:
        # Display the image with the mask
        masked_img = cv2.bitwise_and(img1, mask)
        # Concatenate images horizontally
        concatenated_image = cv2.hconcat([masked_img, image2])
        cv2.imshow("Title of Popup Window", concatenated_image)

        # Wait for 'enter' key to exit loop
        if cv2.waitKey(3) == 13:
            break
    # Save the mask
    cv2.imwrite("../Masks/mask1.png", mask)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
