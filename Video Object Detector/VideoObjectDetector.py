from ultralytics import YOLO
import cv2
import cvzone
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/cars video.mp4")
mask = cv2.imread("../Masks/mask.png")

detector = YOLO("../Models/yolov8n.pt")

limits = [500, 510, 1370, 497]
totalCount = []

# Tracking and counting objects
tracker = Sort(max_age=20)

while True:
    success, image = cap.read()
    imageRegion = cv2.bitwise_and(image, mask)

    results = detector(imageRegion, stream=True)
    detections = np.empty((0, 5))

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Finding coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence in percentages
            confidence = float(box.conf[0])
            roundedConfidence = round(confidence, 2)
            # Finds classId and names
            classNames = int(box.cls[0])
            # Detects only person(0) and cars(2)
            if classNames == 2 and roundedConfidence > 0.30:
                # # Drawing lines
                # cvzone.cornerRect(image, (x1, y1, w, h))
                # # Drawing confidence level and className
                # cvzone.putTextRect(image, f"{detector.names.get(classNames)},{roundedConfidence}",
                #                    (max(0, x1), max(30, y1)),
                #                    thickness=2,
                #                    scale=0.7,
                #                    colorT=(0, 0, 0),
                #                    colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_DUPLEX)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    trackerResult = tracker.update(detections)
    # Drawing line
    cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 191, 255), thickness=5)

    for result in trackerResult:
        x1, y1, x2, y2, trackerId = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, w, h))
        cvzone.putTextRect(image, f"{trackerId}",
                           (max(0, x1), max(30, y1)),
                           thickness=2,
                           scale=0.7,
                           colorT=(0, 0, 0),
                           colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_DUPLEX)

        # Finds centre of cars
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(image, (cx, cy), radius=10, color=(0, 0, 255), thickness=3)
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(trackerId) == 0:
                totalCount.append(trackerId)
        cvzone.putTextRect(image, f'{len(totalCount)}', (400, 80), border=5, thickness=5, font=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Image", image)
    # cv2.imshow("ImageRegion", imageRegion)
    cv2.waitKey(1)
