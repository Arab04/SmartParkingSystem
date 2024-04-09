import numpy as np
import imutils
import easyocr
import cv2
from matplotlib import pyplot as plt


def detectAnprFromImage(path):
    print("Hello")
    # Converts colorful image to black and white image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    # Draws whole picture of image
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise Reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

    keyPoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keyPoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Detects ANPR location and returns location coordinates
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # Mask behind of a ANPR
    mask = np.zeros(gray.shape, dtype="uint8")  # Creating black mask
    new_img = cv2.drawContours(mask, [location], 0, 255, -1)  # Telling which part not to include
    new_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("name", new_img)
    cv2.waitKey(0)

    # Crop the ANPR and get coordinates x1, y1, x2, y2
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_img = gray[x1:x2 + 1, y1:y2 + 1]
    print(f'{x1},{y1},{x2},{y2}')
    # Convert ANPR to String
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped_img)
    print(result)


def takeScreenshot(image, x1, y1, x2, y2):
    carCoordinates = image[y1:y2, x1:x2]
    cv2.imwrite(f"../ScreenShots/{carCoordinates}.jpg", carCoordinates)  # Save the screenshot
    return f'../ScreenShots/{carCoordinates}.jpg'


def methodTwo():
    img = cv2.imread("ScreenShots/image4.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result)


if __name__ == "__main__":
    detectAnprFromImage()
