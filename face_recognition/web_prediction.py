import os
import pickle

import cv2
import numpy as np

import face_recognition.api as face_recognition


def predict():
    def get_full_path(filename):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(filename)]:
                return os.path.join(dirpath, filename)
        return None

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.

    path = get_full_path('face_encodings.data')
    with open(path, 'rb') as filehandle:
        known_face_encodings = pickle.load(filehandle)

    path_counter = get_full_path('encodings_counter.data')
    with open(path_counter, 'rb') as filehandle:
        counter = pickle.load(filehandle)

    names = []
    for k, v in counter.items():
        names.append(k)

    known_face_names = []
    for name in names:
        for j in range(counter[name]):
            known_face_names.append(name)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    faces = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                # print(best_match_index)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name_ in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name_, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            faces.append(name_.lower())
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return faces
