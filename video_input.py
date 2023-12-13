from ultralytics import YOLO
import detection_processing as dp
import cv2

# path to video source
VIDEO_SOURCE = '/home/adrian/Desktop/Capstone/IMG_1543.MOV'
model = YOLO('./best.pt')

cap = cv2.VideoCapture(VIDEO_SOURCE)

process = dp.detection_process()

# open video source
while cap.isOpened():

    success, frame = cap.read()

    if success:
        # uses YOLOv8 tracking in order to track objects
        results = model.track(frame, persist=True, conf=.7, verbose = False)
        # updates firestore if there are new objects being tracked
        process.update_detections(results[0].tojson())
        
        # displays annotated frame
        annotated_frame = results[0].plot()
        imS = cv2.resize(annotated_frame, (1280,720))
        cv2.imshow("Video Soruce", imS)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()