import numpy as np

face_encodings = np.array([[1, 1, 3],
                           [2, 2, 2],
                           [3, 3, 5]])
face_to_compare = np.array([[1],
                            [2],
                            [3]]
                           )


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


print(face_distance(face_encodings, face_to_compare))
print(np.linalg.norm(face_encodings) - np.linalg.norm(face_to_compare))
print(list(face_distance(face_encodings, face_to_compare) <= 3.5))
