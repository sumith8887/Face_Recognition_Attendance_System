import cv2
import face_recognition
import pickle

data = pickle.load(open("C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\encodings\\encodings.pkl","rb"))

video = cv2.VideoCapture(0)

process_frame = True

while True:
    ret, frame = video.read()

    # Resize frame for speed
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    if process_frame:

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb,boxes)

        names = []

        for encoding in encodings:

            matches = face_recognition.compare_faces(
                data["encodings"], encoding
            )

            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = data["names"][index]

            names.append(name)

    process_frame = not process_frame

    cv2.imshow("Fast Recognition",frame)

    if cv2.waitKey(1)==27:
        break

video.release()
cv2.destroyAllWindows()