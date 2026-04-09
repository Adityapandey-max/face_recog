import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'images'
images = []
classNames = []
name_map = {
    "aditya": "Aditya Pandey",
    "alakh": "Alakh Pandey",
    "virat kohli": "Virat Kohli"
}

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)

        # Clean file name
        name = os.path.splitext(cl)[0]
        name = name.replace("_", " ").lower()

        # Apply custom mapping
        if name in name_map:
            name = name_map[name]

        classNames.append(name)

def findEncodings(images):
    encodeList = []
    validNames = []

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        # Skip if no face found
        if len(encodes) > 0:
            encodeList.append(encodes[0])
            validNames.append(classNames[i])

    return encodeList, validNames

def markAttendance(name):
    try:
        with open('attendance.csv', 'a+') as f:
            f.seek(0)
            dataList = f.readlines()
            nameList = []

            for line in dataList:
                entry = line.split(',')
                nameList.append(entry[0])

            if name.upper() not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.write(f'{name.upper()},{dtString}\n')

    except PermissionError:
        print("❌ attendance.csv close karo pehle!")

# Encode faces
encodeListKnown, classNames = findEncodings(images)
print("Encoding Complete...")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if len(encodeListKnown) > 0:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            # ✅ VERIFIED
            if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                name = classNames[matchIndex]

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"{name.upper()} - VERIFIED", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                markAttendance(name)

            else:
                # ❌ NOT VERIFIED
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img, "NOT VERIFIED", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        else:
            cv2.putText(img, "NO DATA", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Face Recognition Attendance', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
