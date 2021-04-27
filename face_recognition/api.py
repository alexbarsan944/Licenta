import PIL.Image
import dlib
import numpy as np
from PIL import ImageFile
from pkg_resources import resource_filename
import os
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

predictor_5_point_model = resource_filename(__name__, "../models/shape_predictor_5_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

predictor_68_point_model = resource_filename(__name__, "../models/shape_predictor_68_face_landmarks.dat")
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

face_recognition_model = resource_filename(__name__, "../models/dlib_face_recognition_resnet_model_v1.dat")
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def rect_to_trbl(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def trbl_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def trim_trbl(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(file, mode='RGB'):
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    return [trim_trbl(rect_to_trbl(face), img.shape) for face in
            _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None, model='5'):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [trbl_to_rect(face_location) for face_location in face_locations]

    if model == '5':
        pose_predictor = pose_predictor_5_point
    else:
        pose_predictor = pose_predictor_68_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def face_landmarks(face_image, face_locations=None):
    landmarks = _raw_face_landmarks(face_image, face_locations, model='5')
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
    return [{
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "left_eye": points[36:42],
        "right_eye": points[42:48]
    } for points in landmarks_as_tuples]


def video_pic_convertor(video_location, output_location):
    # TODO : add face detection to output
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_location}/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success, count)
        count += 1


def get_known_people_from_dataset():
    directory = resource_filename(__name__, '../dataset')
    temp = []
    name_list = []
    for subdir, dirs, files in os.walk(directory):
        temp = dirs
        break
    for name in temp:
        if name == '__pycache__' or name == 'data' or name == '.DS_Store':
            pass
        else:
            name_list.append(name)
    return name_list


# print(get_known_people_from_dataset())


def get_known_people_from_encodings():
    def unnest(d):
        from pandas.io.json._normalize import nested_to_record
        return nested_to_record(d, sep='_')

    import pickle
    people = []
    abs_path = resource_filename(__name__, 'encodings_counter.data')
    # abs_path = os.path.abspath('face_recognition/encodings_counter.data')
    with open(abs_path, 'rb') as filehandle:
        # read the data as binary data stream
        objs = []
        while 1:
            try:
                objs.append(pickle.load(filehandle))
            except EOFError:
                break
    counter = unnest(objs)
    for k in counter[0]:
        people.append(k)
    return people


# print(get_known_people_from_encodings())


def get_path_form_name(name):
    def get_full_path(filename):
        path = (os.path.expanduser('~/Documents/GitHub/Licenta'))
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(filename)]:
                return os.path.join(dirpath, filename)
        return None

    known_people = get_known_people_from_encodings()
    if name not in known_people:
        return None
    else:
        abs_path = get_full_path(name)
        print(abs_path)
    return abs_path

# get_path_form_name('Alex')
