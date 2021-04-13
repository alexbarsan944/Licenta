import face_recognition.api as face_recognition
import cv2
import numpy as np
import pickle
import os
from pathlib import Path


def produce_videos(person='all'):
    if person != 'all':
        # TODO check if name exists in database folder
        video = f"{person}.mp4"

        people = face_recognition.get_people()
        for idx, p in enumerate(people):
            people[idx] = p.lower()

        input_movie = cv2.VideoCapture(f"videos/known/{video}")

        if person not in people:
            input_movie = cv2.VideoCapture(f"videos/unknown/{video}")


        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        wid, leng = input_movie.get(3), input_movie.get(4)

        output_movie = cv2.VideoWriter(f'output/{video}', fourcc, 29.97, (int(wid), int(leng)))

        with open('../face_recognition/face_encodings.data', 'rb') as filehandle:
            known_face_encodings = pickle.load(filehandle)

        names = face_recognition.get_people()
        known_face_names = []
        for name in names:
            for j in range(len(known_face_encodings) // len(names)):
                known_face_names.append(name)

        frame_number = 0

        while True:
            ret, frame = input_movie.read()
            frame_number += 1
            # if frame is not None:
            #     frame = cv2.resize(frame, (frame.shape[0], frame.shape[1]))
            # Quit when the input video file ends
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            output_movie.write(frame)

        # All done!
        input_movie.release()
        cv2.destroyAllWindows()
    else:
        for root, dirs, files in os.walk('../tests/videos'):
            for filename in files:
                if filename != '.DS_Store':
                    print(filename)
                    video = os.path.join(root, filename)
                    video = (os.path.abspath(video))

                    input_movie = cv2.VideoCapture(video)
                    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    wid, leng = input_movie.get(3), input_movie.get(4)

                    output_movie = cv2.VideoWriter(f'output/{video}', fourcc, 29.97, (int(wid), int(leng)))

                    with open('../face_recognition/face_encodings.data', 'rb') as filehandle:
                        known_face_encodings = pickle.load(filehandle)

                    names = face_recognition.get_people()
                    known_face_names = []
                    for name in names:
                        for j in range(len(known_face_encodings) // len(names)):
                            known_face_names.append(name)

                    # Initialize some variables
                    # Initialize some variables
                    face_locations = []
                    face_names = []
                    frame_number = 0

                    while True:
                        # Grab a single frame of video
                        ret, frame = input_movie.read()
                        frame_number += 1

                        # Quit when the input video file ends
                        if not ret:
                            break

                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                        rgb_frame = frame[:, :, ::-1]

                        # Find all the faces and face encodings in the current frame of video
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        face_names = []
                        for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            face_names.append(name)

                        # Label the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            if not name:
                                continue

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                        # Write the resulting image to the output video file
                        print("Writing frame {} / {}".format(frame_number, length))
                        output_movie.write(frame)

                    # All done!
                    input_movie.release()
                    cv2.destroyAllWindows()


# produce_videos(person='alex')
# produce_videos(person='costiS')
# produce_videos(person='costi')
# produce_videos(person='stefan')
# produce_videos(person='raluca')
produce_videos(person='stockvideo2')
produce_videos(person='stockvideo')
