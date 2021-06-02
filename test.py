import os
import pickle

import promptlib

from face_recognition.api import get_known_people_from_encodings, get_known_people_from_dataset
from face_recognition.web_prediction import predict
from test.video_prediction import produce_video

# method = input('Test by:\n (1) Webcam\n(2) video')
method = '2'
output_faces = []
person_name = ''

if method != '1' and method != '2':
    exit(1)
elif method == '1':
    name = input("Enter your name: ")
    faces = predict()
    names_in_db = get_known_people_from_dataset()
    x = {}
    print(faces)
    for nume in names_in_db:
        x[nume] = (faces.count(nume))
    x['unknown'] = faces.count('unknown')
    print(x)
    keys = [j for j in x.keys()]
    values = [j for j in x.values()]

    acc = x[name] * 100 / sum(values)

    exit(5)
elif method == '2':
    prompter = promptlib.Files()
    abs_path_to_video = prompter.file()
    print(abs_path_to_video.split('/')[-2])
    should_be = abs_path_to_video.split('/')[-2]
    file_name = abs_path_to_video.split('.')[0].split('/')[-1]
    output_faces = produce_video(abs_path_to_video, should_be)
    prompter.dst()

    print('Done.')


def check_if_safe(should_be, face_list, frame_no):
    def all_different_than(person, lst):
        for f in lst:
            if f == person:
                return False
        return True

    queue = []
    for face in face_list:
        queue.append(face)
        if len(queue) >= frame_no:
            if all_different_than(should_be, queue):
                return False
            queue.pop(0)
    return True


def return_results(output_faces, should_be):
    for i in range(len(output_faces)):
        output_faces[i] = output_faces[i].lower()

    print(should_be, output_faces)
    safe = check_if_safe(should_be, output_faces, 30)

    if len(output_faces) == 0:
        print('No faces found')
        output_faces = []

    acc = output_faces.count(should_be) * 100 / len(output_faces)
    print(acc)

    known_encodings = get_known_people_from_encodings()
    for i in range(len(known_encodings)):
        known_encodings[i] = known_encodings[i].lower()

    names = {}
    for name in known_encodings:
        names[name] = output_faces.count(name.lower())
    names['unknown'] = output_faces.count('unknown')
    return [file_name.lower(), names, acc, safe]


list = []
if len(output_faces) == 0:
    list[1] = 'noface'
else:
    list = return_results(output_faces, should_be)

if len(output_faces) == 0:
    list[1] = 'noface'

if not os.path.isdir(f'results/{should_be}'):
    try:
        os.mkdir(f'results/{should_be}')
    except OSError:
        print("Creation of the directory %s failed" % should_be)

with open(f'results/{should_be}/{list[0]}.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(list, filehandle)
