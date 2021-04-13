import cv2


def video_pic_convertor(video_location, output_location):
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_location}/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success, count)
        count += 1

    pass


video_location = '/Users/alexandrubarsan/Documents/GitHub/Licenta/face_recognition/train_videos/costi-train.mp4'

video_pic_convertor(video_location, 'train_pictures')
