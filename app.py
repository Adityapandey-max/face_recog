import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

st.title("Face Recognition Attendance System")

path = 'images'
images = []
classNames = []

name_map = {
    "aditya": "Aditya Pandey",
    "alakh": "Alakh Pandey",
    "virat kohli": "Virat Kohli"
}

# Load images
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)

        name = os.path.splitext(cl)[0]
        name = name.replace("_", " ").lower()

        if name in name_map:
            name = name_map[name]

        classNames.append(name)

# Encode faces
def findEncodings(images):
    encodeList = []
    validNames = []

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])
            validNames.append(classNames[i])

    return encodeList, validNames

# Attendance
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
        st.error("Close attendance.csv first!")

encodeListKnown, classNames = findEncodings(images)

st.success("Encoding Complete ✅")

# UI OPTION
option = st.sidebar.selectbox("Choose Option", ["Upload Image", "Use Camera"])

# 📤 IMAGE UPLOAD
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

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

                if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                    name = classNames[matchIndex]

                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, f"{name.upper()} - VERIFIED", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    markAttendance(name)
                    st.success(f"{name} Verified ✅")

                else:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(img, "NOT VERIFIED", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    st.error("Not Verified ❌")

        st.image(img, channels="BGR")

# 📸 CAMERA INPUT
elif option == "Use Camera":
    img_file = st.camera_input("Take a Picture")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

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

                if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                    name = classNames[matchIndex]

                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, f"{name.upper()} - VERIFIED", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    markAttendance(name)
                    st.success(f"{name} Verified ✅")

                else:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(img, "NOT VERIFIED", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    st.error("Not Verified ❌")

        st.image(img, channels="BGR")