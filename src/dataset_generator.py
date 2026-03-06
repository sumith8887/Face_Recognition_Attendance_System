import cv2
import os

student_name = input("Enter student name: ")

dataset_path = f"C:\\Users\\sumit\\OneDrive\\Documents\\Desktop\\23B81A66J2\\Projects\\Face_recognition_for_attendance\\dataset\\{student_name}"

# Create folder if not exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Count existing images
existing_images = len(os.listdir(dataset_path))
count = existing_images

cam = cv2.VideoCapture(0)

print("Press 'C' to capture image")
print("Press 'Q' to quit")

while True:

    ret, frame = cam.read()

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):

        img_name = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(img_name, frame)

        print(f"Image {count} saved")
        count += 1

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()