import os
import cv2
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

dataset_path = r"C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\dataset"

mtcnn = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = []
names = []

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = mtcnn(rgb)

        if face is not None:

            embedding = model(face.unsqueeze(0)).detach().numpy()[0]

            embeddings.append(embedding)
            names.append(person)

data = {
    "embeddings": embeddings,
    "names": names
}

os.makedirs("C:/Users/sumit/OneDrive/Documents/Desktop/23B81A66J2/Projects/Face_recognition_for_attendance/encodings", exist_ok=True)

with open("C:/Users/sumit/OneDrive/Documents/Desktop/23B81A66J2/Projects/Face_recognition_for_attendance/encodings/embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully")