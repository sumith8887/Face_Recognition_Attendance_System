import pickle
from sklearn.neighbors import KNeighborsClassifier

with open("C:/Users/sumit/OneDrive/Documents/Desktop/23B81A66J2/Projects/Face_recognition_for_attendance/encodings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

X = data["embeddings"]
y = data["names"]

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)

with open("C:/Users/sumit/OneDrive/Documents/Desktop/23B81A66J2/Projects/Face_recognition_for_attendance/encodings/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("KNN model saved")