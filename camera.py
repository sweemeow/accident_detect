import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import winsound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


def startapplication():
    # for camera use video = cv2.VideoCapture(0)
    video = cv2.VideoCapture('meow.mp4')
    count = 0
    while True:
        ret, frame = video.read()
        try:
            count = count+1
            print(f"processing frame number {count}")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            break

        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if (pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))

            # to beep when alert:
            if (prob > 97):
                print("accident")
                frequency = 2500 
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
            else: 
                print("not accident")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob),
                        (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)


if __name__ == '__main__':
    startapplication()
