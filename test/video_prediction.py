import os
import pickle

import cv2
import numpy as np

import face_recognition.api as face_recognition


def produce_video(abs_path_to_video, should_be):
    def get_full_path(filename):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(filename)]:
                return os.path.join(dirpath, filename)
        return None

    def get_full_path_name(filename):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.startswith(filename)]:
                if 'known' in dirpath or 'unknown' in dirpath:
                    return os.path.join(dirpath, filename)
        return None

    def get_full_path_dir(dir):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            if dir in dirnames:
                return os.path.join(dirpath, dir)
        return None

    movie_path = abs_path_to_video
    input_movie = cv2.VideoCapture(movie_path)

    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    wid, leng = input_movie.get(3), input_movie.get(4)

    output_path = get_full_path_dir('output')
    person_name = abs_path_to_video.split('.')[0].split('/')[-1]

    output_movie = cv2.VideoWriter(f'{output_path}/{person_name}.mp4', fourcc, 29.97, (int(wid), int(leng)))

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
    right_prediction = 0
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        frame_number += 1
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
            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name_, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            faces.append(name_.lower())

            if should_be.lower() == name_.lower():
                right_prediction += 1
            acc = right_prediction * 100 / len(faces)

            if frame_number % 10 == 0:
                print("Writing frame {} / {}".format(frame_number, length), f"Accuracy={acc}%")

        # Display the resulting image
        # output_movie.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return faces

# produce_video(person='alex')
# produce_video(person='costiS')
# produce_video(person='costi')
# produce_video(person='stefan')
# produce_video('raluca')
# produce_video(person='stockvideo2')
# produce_video('')

# def produce_videos(person='all'):
#     if person != 'all':
#         produce_video(person)
#     else:
#         for root, dirs, files in os.walk('../tests/videos'):
#             for filename in files:
#                 if filename != '.DS_Store':
#                     print(filename)
#                     video = os.path.join(root, filename)
#                     video = (os.path.abspath(video))
#
#                     input_movie = cv2.VideoCapture(video)
#                     length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
#
#                     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                     wid, leng = input_movie.get(3), input_movie.get(4)
#
#                     output_movie = cv2.VideoWriter(f'output/{video}', fourcc, 29.97, (int(wid), int(leng)))
#
#                     with open('../face_recognition/face_encodings.data', 'rb') as filehandle:
#                         known_face_encodings = pickle.load(filehandle)
#
#                     names = face_recognition.get_known_people()
#                     known_face_names = []
#                     for name in names:
#                         for j in range(len(known_face_encodings) // len(names)):
#                             known_face_names.append(name)
#
#                     # Initialize some variables
#                     # Initialize some variables
#                     face_locations = []
#                     face_names = []
#                     frame_number = 0
#
#                     while True:
#                         # Grab a single frame of video
#                         ret, frame = input_movie.read()
#                         frame_number += 1
#
#                         # Quit when the input video file ends
#                         if not ret:
#                             break
#
#                         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#                         rgb_frame = frame[:, :, ::-1]
#
#                         # Find all the faces and face encodings in the current frame of video
#                         face_locations = face_recognition.face_locations(rgb_frame)
#                         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#                         face_names = []
#                         for face_encoding in face_encodings:
#                             # See if the face is a match for the known face(s)
#                             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                             name = "Unknown"
#                             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                             best_match_index = np.argmin(face_distances)
#
#                             if matches[best_match_index]:
#                                 name = known_face_names[best_match_index]
#
#                             face_names.append(name)
#
#                         # Label the results
#                         for (top, right, bottom, left), name in zip(face_locations, face_names):
#                             if not name:
#                                 continue
#
#                             # Draw a box around the face
#                             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#                             # Draw a label with a name below the face
#                             cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
#                             font = cv2.FONT_HERSHEY_DUPLEX
#                             cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
#
#                         # Write the resulting image to the output video file
#                         print("Writing frame {} / {}".format(frame_number, length))
#                         output_movie.write(frame)
#
#                     # All done!
#                     input_movie.release()
#                     cv2.destroyAllWindows()
