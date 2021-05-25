import os

import cv2
import dlib
from pkg_resources import resource_filename


def get_pictures(name):
    # Load the detector
    detector = dlib.get_frontal_face_detector()


    # read the image
    cap = cv2.VideoCapture(0)
    total = 0

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
            # if y1 >= 100 >= y2:
            #     face_frame = frame[y1 - 100:y2 + 100, x1 - 100:x2 + 100]
            # else:
            #     face_frame = frame[y1 - 50:y2 + 50, x1 - 50:x2 + 50]

            face_frame = frame[int(0.8 * y1): int(1.2 * y2), int(0.8 * x1): int(x2 * 1.2)]

            # show the image

            cv2.imshow(winname="Face", mat=frame)
            save_location = resource_filename(__name__, "../dataset/" + name)
            p = os.path.sep.join([save_location + '/', "{}.png".format(str(total).zfill(3))])
            cv2.imwrite(p, face_frame)
            total += 1
            if total % 10 == 0:
                print(total)

        # Exit when escape is pressed
        if cv2.waitKey(delay=1) == 27 or total > 500:
            break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()
