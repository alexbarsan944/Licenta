import os

import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture(0)
total = 0
name = input('Enter your name: ')
path = os.getcwd()
try:
    os.mkdir(path + '/' + name)
except OSError:
    print("Creation of the directory %s failed" % name)
    exit(1)
else:
    print("Successfully created the directory %s " % name)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Crop the face
        if y1 >= 100 >= y2:
            face_frame = frame[y1 - 100:y2 + 100, x1 - 100:x2 + 100]
        else:
            face_frame = frame[y1 - 50:y2 + 50, x1 - 50:x2 + 50]
        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # show the image

        cv2.imshow(winname="Face", mat=face_frame)

        p = os.path.sep.join([name + '/', "{}.png".format(str(total).zfill(3))])
        cv2.imwrite(p, face_frame)
        total += 1
        print(total)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()
