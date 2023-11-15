from ultralytics import YOLO
import detection_processing as dp
import cv2


VIDEO_SOURCE = '/home/adrian/Desktop/Capstone/Los Angeles In The Streets - Episode 4.mp4'

model = YOLO('/home/adrian/Desktop/Capstone/mobile_end/best.pt')

cap = cv2.VideoCapture(VIDEO_SOURCE)

process = dp.detection_process()

while cap.isOpened():

    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, conf=.3, verbose = False)
        process.update_detections(results[0].tojson())

        annotated_frame = results[0].plot()

        cv2.imshow("My model inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()