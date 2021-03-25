import PIL.Image
import dlib
import numpy as np
from PIL import ImageFile
from pkg_resources import resource_filename

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
    landmarks = _raw_face_landmarks(face_image, face_locations, model='68')
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
    return [{
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "left_eye": points[36:42],
        "right_eye": points[42:48]
    } for points in landmarks_as_tuples]
