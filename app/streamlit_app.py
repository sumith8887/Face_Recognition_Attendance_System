import streamlit as st
import cv2
import pickle
import numpy as np
import tempfile
import torch
import pandas as pd
import os
from datetime import datetime
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN

# create attendance for current session

# unique file for each app run
run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
attendance_file = f"attendance/attendance_{run_time}.csv"

# enter period name 
period_name = st.text_input("Enter Period Name (Example: DL, AINN, BEFA)")

st.title("🎓 Face Recognition Attendance from Video")

st.write("Upload a classroom video to recognize students and mark attendance.")

# ----------------------------
# Load Models
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@st.cache_resource
def load_knn():
    with open(r"C://Users//sumit//OneDrive//Documents//Desktop//23B81A66J2//Projects//Face_recognition_for_attendance//encodings//knn_model.pkl","rb") as f:
        return pickle.load(f)

knn = load_knn()

# ----------------------------
# Attendance Function
# ----------------------------
import pandas as pd
import os
from datetime import datetime

def mark_attendance(name, period):

    today = datetime.now().strftime("%Y-%m-%d")

    os.makedirs("attendance", exist_ok=True)

    file_path = f"attendance/attendance_{today}.csv"

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name"])

    # Insert period column before Attendance %
    if period not in df.columns:

        if "Attendance %" in df.columns:
            insert_position = df.columns.get_loc("Attendance %")
            df.insert(insert_position, period, "")
        else:
            df[period] = ""

    # Mark attendance
    if name in df["Name"].values:

        df.loc[df["Name"] == name, period] = "Present"

    else:

        new_row = {"Name": name, period: "Present"}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(file_path, index=False)

    # update attendance percentage
    update_attendance_percentage(file_path)

# ----------------------------
# update attendance % 
# ----------------------------

def update_attendance_percentage(file_path):

    df = pd.read_csv(file_path)

    period_columns = [col for col in df.columns if col not in ["Name", "Attendance %"]]

    total_periods = len(period_columns)

    percentages = []

    for _, row in df.iterrows():

        present_count = 0

        for col in period_columns:
            if row[col] == "Present":
                present_count += 1

        if total_periods > 0:
            percentage = (present_count / total_periods) * 100
        else:
            percentage = 0

        percentages.append(round(percentage,2))

    df["Attendance %"] = percentages

    df.to_csv(file_path, index=False)

# ----------------------------
# color the attendance % column
# ----------------------------
def color_attendance(val):

    if val >= 75:
        color = "green"

    elif val >= 50:
        color = "orange"

    else:
        color = "red"

    return f"color: {color}; font-weight: bold"

# ----------------------------
# Upload Video
# ----------------------------

uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

recognized_students = set()

if uploaded_video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_window = st.image([])

    st.write("Processing video...")

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Skip frames for speed
        if frame_count % 5 != 0:
            continue

        # Resize frame (faster detection)
        frame = cv2.resize(frame,(640,360))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, _ = detector.detect(rgb)

        if boxes is not None:

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1)
                y1 = max(0, y1)

                face_img = rgb[y1:y2, x1:x2]

                try:
                    face_img = cv2.resize(face_img, (160,160))
                except:
                    continue

                face_tensor = torch.tensor(face_img).permute(2,0,1).float()/255

                embedding = facenet(face_tensor.unsqueeze(0)).detach().numpy()

                distances, indices = knn.kneighbors(embedding)

                if distances[0][0] < 0.6:
                    name = knn.predict(embedding)[0]
                    recognized_students.add(name)
                    mark_attendance(name, period_name)
                else:
                    name = "Unknown"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,name,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,255,0),2)

        frame_window.image(frame, channels="BGR")

    cap.release()

    st.success("Video processing completed!")

# ----------------------------
# Show Recognized Students
# ----------------------------

st.subheader("Recognized Students")

if len(recognized_students) > 0:

    for student in recognized_students:
        st.write("✅", student)

else:
    st.write("No registered students detected.")

# ----------------------------
# Display Attendance Table
# ----------------------------
today = datetime.now().strftime("%Y-%m-%d")

attendance_file = f"attendance/attendance_{today}.csv"

if os.path.exists(attendance_file):

    # Check if file is empty
    if os.path.getsize(attendance_file) == 0:

        df = pd.DataFrame(columns=["Name"])

    else:
        try:
            df = pd.read_csv(attendance_file)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name"])

    st.subheader("Today's Attendance")

    if "Attendance %" in df.columns:

        def color_attendance(val):

            if val >= 75:
                color = "green"
            elif val >= 50:
                color = "orange"
            else:
                color = "red"

            return f"color:{color}; font-weight:bold"

        styled_df = df.style.map(color_attendance, subset=["Attendance %"])

        st.dataframe(styled_df)

    else:
        st.dataframe(df)

    with open(attendance_file,"rb") as f:

        st.download_button(
            "Download Attendance",
            f,
            file_name=f"attendance_{today}.csv"
        )

else:

    st.info("No attendance recorded yet.")