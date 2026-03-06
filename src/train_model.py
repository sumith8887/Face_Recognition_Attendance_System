import face_recognition
import os
import pickle
import cv2

dataset_path = "C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\dataset"

known_encodings = []
known_names = []

print("Training started...")

for person in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:

            known_encodings.append(encoding)
            known_names.append(person)

print("Total faces trained:", len(known_encodings))

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\encodings\\encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Training completed successfully")