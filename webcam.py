import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("mask_model.keras")

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face = frame[y:y+height, x:x+width]

            if face.size != 0:
                face_resized = cv2.resize(face, (224, 224))
                face_array = face_resized / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                prediction = model.predict(face_array)

                if prediction[0][0] > 0.5:
                    label = "Mask"
                    color = (0, 255, 0)
                else:
                    label = " No Mask"
                    color = (0, 0, 255)

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)

    cv2.imshow("Mask Detection - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()