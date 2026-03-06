import cv2
import face_recognition
import pickle
from mark_attendance import mark_attendance

data = pickle.load(open("C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\data\\encoded_faces\\encodings.pkl", "rb"))

video = cv2.VideoCapture("C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\data\\videos\\video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):

        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = data["names"][index]
            mark_attendance(name)

        top, right, bottom, left = box

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()