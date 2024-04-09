from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import numpy as np
from AnprDetection import plateDetection as pltd


def initialize():
    # Initialize video capture
    cap = cv2.VideoCapture("../Videos/cars video.mp4")
    # Initialize mask
    mask = cv2.imread("../Masks/mask.png")
    # Initialize detector
    detector = YOLO("../Models/yolov8n.pt")
    # Initialize tracker
    tracker = Sort(max_age=20)
    return cap, mask, detector, tracker


def process_frame(cap, mask, detector, tracker):
    success, image = cap.read()
    # Put Mask to the Video
    imageRegion = cv2.bitwise_and(image, mask)

    results = detector(imageRegion, stream=True)
    # For getting x1, y1, x2, y2, trackerId
    detections = np.empty((0, 5))
    # Get
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(box.conf[0])
            roundedConfidence = round(confidence, 2)
            classNames = int(box.cls[0])
            if classNames == 2 and roundedConfidence > 0.30:
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))
                path = pltd.takeScreenshot(image, x1, y1, x2, y2)
                print(path)
                #pltd.detectAnprFromImage(path)
    trackerResult = tracker.update(detections)
    return image, trackerResult


def draw(image, trackerResult, limits, totalCount):
    # Draws line: if car crosses this line, car will be detected
    #cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 191, 255), thickness=5)

    for result in trackerResult:
        x1, y1, x2, y2, trackerId = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, w, h))
        cvzone.putTextRect(image, f"{trackerId}", (max(0, x1), max(30, y1)),
                           thickness=2,
                           scale=0.7,
                           colorT=(0, 0, 0),
                           colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_DUPLEX)
        cx, cy = x1 + w // 2, y1 + h // 2
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(trackerId) == 0:
                car_roi = image[y1:y2, x1:x2]
                cv2.imwrite(f"../ScreenShots/car_{trackerId}.jpg", car_roi)  # Save the screenshot
                totalCount.append(trackerId)
        cvzone.putTextRect(image, f'Total Count: {len(totalCount)}', (100, 50), border=10, colorR=(255, 0, 0),
                           colorT=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.5)

    return image


def main():
    cap, mask, detector, tracker = initialize()
    # Needs to be changed to autonomous
    limits = [500, 510, 1370, 497]
    totalCount = []
    while True:
        image, trackerResult = process_frame(cap, mask, detector, tracker)
        image = draw(image, trackerResult, limits, totalCount)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
