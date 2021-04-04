import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from PyQt5 import QtWidgets
from keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSlot



class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Image Analysis Device'
        self.left = 700
        self.top = 300
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedWidth(700)
        self.setFixedHeight(300)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label = QtWidgets.QLabel(self)
        label1 = QtWidgets.QLabel(self)
        label1.setText("TO")
        label2 = QtWidgets.QLabel(self)
        label2.setText("IMAGE ANALYSIS HUB!!!")
        label.setText("WELCOME")
        label.move(270, 50)
        label1.move(320, 85)
        label2.move(160, 110)
        label.setFont(QFont('Lucida Console', 20))
        label1.setFont(QFont('Lucida Console', 12))
        label2.setFont(QFont('Lucida Console', 20))
        button1 = QPushButton('FACE DETECTION', self)
        button2 = QPushButton('FACE RECOGNITION', self)
        button3 = QPushButton('EMOTION DETECTION', self)
        button = QPushButton('FACE_MASK', self)
        button.setToolTip('This is an example button')
        button.setGeometry(70, 200,100,50)
        button1.setGeometry(180, 200,120,50)
        button2.setGeometry(310, 200,130,50)
        button3.setGeometry(450, 200,140,50)


        button.clicked.connect(self.on_click)
        button1.clicked.connect(self.f)

        self.show()

    @pyqtSlot()
    def f(self):
        video = cv2.VideoCapture(0)

        facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        count = 0

        while True:
            ret, frame = video.read()
            faces = facedetect.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in faces:
                count = count + 1
                name = './images/face_without_mask/' + str(count) + '.jpg'

                cv2.imwrite(name, frame[y:y + h, x:x + w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imshow("WindowFrame", frame)
            cv2.waitKey(1)
            if count > 500:
                break
        video.release()
        cv2.destroyAllWindows()
    def on_click(self):
        facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        threshold = 0.30
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        font = cv2.FONT_HERSHEY_COMPLEX
        model = load_model('MyTrainingModel.h5')

        def preprocessing(img):
            img = img.astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255
            return img

        def get_className(classNo):
            if classNo == 0:
                return "Mask"
            elif classNo == 1:
                return "No Mask"

        while True:
            sucess, imgOrignal = cap.read()
            faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
            for x, y, w, h in faces:
                crop_img = imgOrignal[y:y + h, x:x + h]
                img = cv2.resize(crop_img, (32, 32))
                img = preprocessing(img)
                img = img.reshape(1, 32, 32, 1)
                prediction = model.predict(img)
                classIndex = model.predict_classes(img)
                probabilityValue = np.amax(prediction)
                if probabilityValue > threshold:
                    if classIndex == 0:
                        cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                        cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif classIndex == 1:
                        cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (50, 50, 255), 2)
                        cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (50, 50, 255), -2)
                        cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Result", imgOrignal)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())