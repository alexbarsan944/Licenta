import os
import pickle
import face_recognition.api as face_recognition
from pkg_resources import resource_filename


def store_face_encodings_all(dataset_count=60):
    known_face_encodings = []
    total = {}
    names = face_recognition.get_known_people_from_dataset()
    print(names)
    for name in names:
        counter = 0
        directory = resource_filename(__name__, "../dataset/" + name + '/')
        print(name)
        for filename in os.listdir(directory):
            if counter < dataset_count:
                if filename != '.DS_Store':
                    image = face_recognition.load_image_file(directory + filename)
                    enc = face_recognition.face_encodings(image)
                    if len(enc) != 0:
                        known_face_encodings.append(enc[0])
                        counter += 1
        total[name] = counter

    face_enc_dir = resource_filename(__name__, '../face_recognition/face_encodings.data')
    face_enc_counter = resource_filename(__name__, '../face_recognition/encodings_counter.data')

    with open(face_enc_dir, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(known_face_encodings, filehandle)

    with open(face_enc_counter, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(total, filehandle)

    with open(face_enc_counter, 'rb') as filehandle:
        # read the data as binary data stream
        counter = pickle.load(filehandle)

    print(counter)


store_face_encodings_all()
