import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import os

os.environ["ALSA_LOG_LEVEL"] = "none"

face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

session = ort.InferenceSession(r"/home/ayush02/models/gaze.onnx")

cap = cv2.VideoCapture(0)

def extract_eye_crop(frame, landmarks):
    h, w = frame.shape[:2]

    if landmarks is None or len(landmarks) < 6:
        return None
    
    left_eye = landmarks[:6]

    if left_eye.size == 0:
        return None
    
    x_coords = left_eye[:, 0]
    y_coords = left_eye[:, 1]

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    x_min = max(0, int(np.min(x_coords)*w))
    y_min = max(0, int(np.min(y_coords)*h))
    x_max = min(w, int(np.max(x_coords)*w))
    y_max = min(h, int(np.max(y_coords)*h))

    if x_max <= x_min or y_max<= y_min:
        return None
    
    crop = frame[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (64, 64))
    return crop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_app.get(frame)

    if faces:
        face = faces[0]
        lm = face.kps

        eye_crop = extract_eye_crop(frame, lm)
        if eye_crop is None:
            continue
        
        eye_input = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        eye_input = eye_input.astype(np.float32) / 255.0
        eye_input = np.transpose(eye_input, (2,0,1))[None]

        gaze = session.run(None, {"input":eye_input})[0]
        gaze_x, gaze_y = gaze[0]


        cv2.putText(frame, f"Gaze Vector: {gaze_x:.2f}, {gaze_y:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
    #cv2.imshow("Gaze Engine", frame)
    print("Gaze Vector:", gaze_x, gaze_y)
    #if cv2.waitKey(1)==27:
        #break
