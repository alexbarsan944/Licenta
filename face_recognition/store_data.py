import os
import pickle
import face_recognition.api as face_recognition

known_face_encodings = []
total = {}
for name in ['Alex', 'Raluca', 'Costi', 'Stefan', 'CostiS']:
    counter = 0
    directory = '/Users/alexandrubarsan/Documents/GitHub/Licenta/dataset/' + name + '/'
    print(name)
    for filename in os.listdir(directory):
        if counter < 500:
            image = face_recognition.load_image_file(directory + filename)
            enc = face_recognition.face_encodings(image)
            if len(enc) != 0:
                known_face_encodings.append(enc[0])
                counter += 1
    total[name] = counter

with open('face_encodings.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(known_face_encodings, filehandle)

with open('encodings_counter.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(total, filehandle)

with open('encodings_counter.data', 'rb') as filehandle:
    # read the data as binary data stream
    counter = pickle.load(filehandle)

print(counter)
