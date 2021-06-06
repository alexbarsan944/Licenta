import os
import pickle

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

import face_recognition.api as face_recognition


def update_main_files_from_folder():
    face_enc_dir = resource_filename(__name__, f'../face_recognition/face_encodings')
    face_enc_counter_dir = resource_filename(__name__, f'../face_recognition/encodings_counter')
    face_enc_file = resource_filename(__name__, f'../face_recognition/face_encodings.data')
    face_enc_counter_file = resource_filename(__name__, f'../face_recognition/encodings_counter.data')

    total = {}

    known_face_encodings = []
    for filename in os.listdir(face_enc_dir):
        #  print(filename)
        with open(face_enc_dir + '/' + filename, 'rb') as filehandle:  # read from directory
            data = pickle.load(filehandle)
            known_face_encodings.append(data)
    l2d = []
    for e1 in known_face_encodings:
        for e2 in e1:
            l2d.append(e2)
    with open(face_enc_file, 'wb') as filehandle:
        pickle.dump(l2d, filehandle)

    for filename in os.listdir(face_enc_counter_dir):
        # print(filename)
        name = filename.split('.data')[0]
        with open(face_enc_counter_dir + '/' + filename, 'rb') as filehandle:
            data2 = pickle.load(filehandle)
            total[name] = data2[name]

    with open(face_enc_counter_file, 'wb') as filehandle:
        pickle.dump(total, filehandle)


def print_main_files():
    face_enc = resource_filename(__name__, f'../face_recognition/face_encodings.data')
    face_enc_counter = resource_filename(__name__, f'../face_recognition/encodings_counter.data')

    with open(face_enc, 'rb') as filehandle:
        data = pickle.load(filehandle)
    with open(face_enc_counter, 'rb') as filehandle:
        data2 = pickle.load(filehandle)
    print(np.array(data).shape)
    print(np.array(data2))
    # df = pd.DataFrame(data=data2, index=data2.values())
    # df.to_csv(f"counter.csv", index=True, header=True)


def store_face_encodings(name, dataset_count=600):
    known_face_encodings = []
    total = {}
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
                    if counter % 10 == 0:
                        print(counter)
        face_enc_dir = resource_filename(__name__, f'../face_recognition/face_encodings/{name}.data')
        with open(face_enc_dir, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(known_face_encodings, filehandle)

    total[name] = counter

    face_enc_counter = resource_filename(__name__, f'../face_recognition/encodings_counter/{name}.data')
    with open(face_enc_counter, 'wb') as filehandle:
        pickle.dump(total, filehandle)

    update_main_files_from_folder()
    print_main_files()


store_face_encodings('Raluca')
store_face_encodings('Stefan')
store_face_encodings('tomsa_s')
store_face_encodings('twin1')
store_face_encodings('twin2')
store_face_encodings('yt1')
store_face_encodings('yt2')
store_face_encodings('yt3')
store_face_encodings('yt4')

# update_main_files_from_folder()
# print_main_files()


def store_face_encodings_all(dataset_count=600):
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
                        if counter % 10 == 0:
                            print(counter)
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

#  store_face_encodings_all()
