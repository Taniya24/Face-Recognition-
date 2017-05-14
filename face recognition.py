import cv2, os, time
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.createLBPHFaceRecognizer()

def image_and_label(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    images=[]
    labels=[]
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        print(image)
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
        faces = faceCascade.detectMultiScale(image)
        for (x,y,w,h) in faces:
            images.append(image[y:y+h,x:x+w])
            labels.append(nbr)
            cv2.imshow("Creating Database...", image[y:y+h,x:x+w])
            cv2.waitKey(2)
    return images,labels
path = 'pro'
images, labels = image_and_label(path)
cv2.destroyAllWindows()
camera_port = 0
ramp_frames = 30
cam = cv2.VideoCapture(camera_port)

def get_img():
    getval, im = cam.read()
    return im

for i in range(ramp_frames):
    temp = get_img()
    print("Capturing Image")

img = get_img()
file = "pro_check//test.jpg"
file1 = "present//test.jpg"
file2 = "absent//test.jpg"
cv2.imwrite(file, img)
recognizer.train(images,np.array(labels))
reco_path = 'pro_check'
person = 15
image_paths = [os.path.join(reco_path,f) for f in os.listdir(reco_path)]
print(image_paths)
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y:y+h, x:x+w])
      #  print(conf)
        if nbr_predicted==person:
            if conf<70.0:
                cv2.imwrite(file1, img)
                print("Person %d is present" %(nbr_predicted))
            else:
                print("Person suspected as subject%d, but we aren't sure. Please try again." %(nbr_predicted))
        else:
            cv2.imwrite(file2, img)
            print("Person Not recognised")
        cv2.waitKey(0)

