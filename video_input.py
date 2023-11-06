from ultralytics import YOLO
import cv2


VIDEO_SOURCE = '/home/adrian/Desktop/Capstone/Los Angeles In The Streets - Episode 4.mp4'

model = YOLO('/home/adrian/Desktop/Capstone/mobile_end/best.pt')

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():

    success, frame = cap.read()

    if success:
        results = model(frame, conf=.3)

        annotated_frame = results[0].plot()

        cv2.imshow("My model inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()