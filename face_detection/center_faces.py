import sys
import dlib
import cv2
import openface
from os.path import isfile, join
from os import listdir

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "/Users/alexandrubarsan/Documents/GitHub/Licenta/models/shape_predictor_68_face_landmarks.dat"

mypath = '../dataset/CostiS'
# Take the image file name from the command line
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for pic_no, file_name in enumerate(onlyfiles):
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    # Take the image file name from the command line
    file_name = '../dataset/CostiS/' + file_name

    # Load the image
    image = cv2.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(128, image, face_rect,
                                         landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        # Save the aligned image to a file
        cv2.imwrite("aligned_faces/CostiS/aligned_face_{}.jpg".format(pic_no), alignedFace)
