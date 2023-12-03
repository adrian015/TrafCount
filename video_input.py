from ultralytics import YOLO
import detection_processing as dp
import cv2


VIDEO_SOURCE = '/home/adrian/Desktop/Capstone/IMG_1543.MOV'

model = YOLO('/home/adrian/Desktop/Capstone/mobile_end/best.pt')

cap = cv2.VideoCapture(VIDEO_SOURCE)

process = dp.detection_process()

while cap.isOpened():

    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, conf=.7, verbose = False)
        process.update_detections(results[0].tojson())

        annotated_frame = results[0].plot()
        imS = cv2.resize(annotated_frame, (1280,720))
        cv2.imshow("My model inference", imS)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()