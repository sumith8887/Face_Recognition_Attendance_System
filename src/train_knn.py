import pickle
from sklearn.neighbors import KNeighborsClassifier

with open("encodings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

X = data["embeddings"]
y = data["names"]

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)

with open("encodings/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("KNN model saved")