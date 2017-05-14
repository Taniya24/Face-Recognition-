import cv2
import sys
import numpy as np

imagePath = sys.argv[1]

Id= input('enter your id')
sampleNum=0
# Create the haar cascade
face_cascade = cv2.CascadeClassifier('C:\\opencv1\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
faceDet3 = cv2.CascadeClassifier('C:\\opencv1\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml')

# Read the image
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def DetectFace(image, faceDet3):

    min_size = (20,20)
    image_scale = 1
    min_neighbors = 3
    haar_flags = 0

    # Allocate the temporary images
    smallImage = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8 ,1)

    # Scale input image for faster processing
    cv.Resize(image, smallImage, cv.CV_INTER_LINEAR)

# Detect faces in the image


faces = face_cascade.detectMultiScale(gray, 1.22, 3, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
print("Found {0} faces!".format(len(faces)))           
print("Found {0} faces!".format(len(face3)))
# Draw a rectangle around the faces


for (x, y, w, h) in face3:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #incrementing sample number 
    sampleNum=sampleNum+1
    #saving the captured face in the dataset folder
    cv2.imwrite("C:\Python36\Face\dataSet1\picture"+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
    
drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_rectangle(event,a,b,flags,param):
    global ix,iy,drawing,sampleNum
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = a,b
                

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image,(ix,iy),(a,b),(0,255,0),2)       
        sampleNum=sampleNum+1
        cv2.imwrite("C:\Python36\Face\dataSet1\picture"+Id +'.'+ str(sampleNum) + ".jpg", gray[iy:b,ix:a])

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)



while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
