import os
import pickle
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

import face_recognition.api as face_recognition

args = {
    "shape_predictor": "models/shape_predictor_68_face_landmarks.dat"
}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


def predict(frames_count=30):
    # START CODE FROM https://github.com/mmenxin/eye-blink-detection-demo

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 2

    COUNTER = 0
    TOTAL = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # STOP CODE FROM https://github.com/mmenxin/eye-blink-detection-demo

    def get_full_path(filename):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(filename)]:
                return os.path.join(dirpath, filename)
        return None

    def check_if_right_name(name, face_list):
        if name.lower() == 'unknown':
            return False
        if len(face_list) < frames_count:
            return False
        else:
            for output in (face_list[-frames_count:]):
                if name.lower() != output.lower():
                    return False
        return True

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
    faces = []
    approved = False
    start_time = time.time()
    total_time = None
    number_of_frames = None

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            # START CODE FROM https://github.com/mmenxin/eye-blink-detection-demo

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                COUNTER = 0

            cv2.putText(
                frame,
                "Blinks: {}".format(TOTAL),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        # STOP CODE FROM https://github.com/mmenxin/eye-blink-detection-demo
        # START CODE FROM https://github.com/ageitgey/face_recognition

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(len(face_encodings))
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
            # STOP CODE FROM https://github.com/ageitgey/face_recognition

            if name == 'unknown':
                approved = False
                TOTAL = 0
                COUNTER = 0

            face_names.append(name)
        if not approved:
            if check_if_right_name(name, face_list=faces) is True and name.lower() != 'unknown' and TOTAL > 1:
                approved = True
                total_time = time.time() - start_time
                number_of_frames = len(faces)

        # Display the results
        if approved is False:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 40), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 10, bottom - 10), 0, 1.1, (255, 255, 255), 1)
                faces.append(name.lower())
        else:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (127, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (127, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f'Approved as {name}', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                faces.append(name.lower())
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return faces, total_time, number_of_frames, approved
