import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

class detection_process:
    def __init__(self):
        self.previous_detections = {'-cyclist' : {-1}, '-large_vehicle' : {-1}, '-motorcyclist' : {-1}, '-pedestrian' : {-1}, '-vehicle' : {-1}}
        cred = credentials.Certificate('firebase_credential_key.json')
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def print_ids(self, new_detections):
        data = json.loads(new_detections)
        for detection in data:
            print(detection['track_id'])

    # Checks set of class type in order to see if track_id has been seen,
    # if so it updates firestore with time stamp and adds it to the set
    def update_detections(self, new_detections):
        
        data = json.loads(new_detections)
        for detection in data:
            if 'track_id' in detection:
                if detection['track_id'] not in self.previous_detections[detection['name']]:
                    print("NEW OBJECT: ", detection['name'], ": ", detection['track_id'])
                    doc_ref = self.db.collection("Traffic")
                    doc_ref.add({"type": detection['name'], "time_detected": firestore.SERVER_TIMESTAMP})
                    self.previous_detections[detection['name']].add(detection['track_id'])