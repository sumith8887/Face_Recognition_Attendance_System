import csv
import os
from datetime import datetime

file = "attendance/attendance.csv"

os.makedirs("attendance", exist_ok=True)

def mark_attendance(name):

    if not os.path.exists(file):

        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name","Date","Time"])

    with open(file, "r+") as f:

        lines = f.readlines()

        names = [line.split(",")[0] for line in lines]

        if name not in names:

            now = datetime.now()

            time = now.strftime("%H:%M:%S")
            date = now.strftime("%Y-%m-%d")

            f.write(f"\n{name},{date},{time}")