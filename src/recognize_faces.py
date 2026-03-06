import cv2
import face_recognition
import pickle
import numpy as np

with open("C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\encodings\\encodings.pkl", "rb") as f:
    data = pickle.load(f)

cam = cv2.VideoCapture(0)

TOLERANCE = 0.45   

while True:

    ret, frame = cam.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)

    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):

        distances = face_recognition.face_distance(data["encodings"], encoding)

        min_distance = np.min(distances)

        name = "Unknown"

        if min_distance < TOLERANCE:

            index = np.argmin(distances)
            name = data["names"][index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.putText(frame, name,
                    (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()