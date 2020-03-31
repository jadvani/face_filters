# Face filters (Snapchat like) using OpenCV
# @author:Javier Advani & Rafa Fernández
#based on @kg777 code with tweepy options and other filters 
import cv2
import logging as log
import datetime as dt
from datetime import datetime
from time import sleep
#import tweepy

i = datetime.now()               #take time and date for filename  
now = i.strftime('%Y%m%d-%H%M%S')

cascPath = "haarcascade_frontalface_default.xml"  # for face detection
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
video_capture.set(3,600)
video_capture.set(4,400)


anterior = 0
mst = cv2.imread('moustache.png')
hat = cv2.imread('cowboy_hat.png')
dog = cv2.imread('dog_filter.png')
flor=cv2.imread('flor2.png')
ieee=cv2.imread('ieee.png')

# =============================================================================
def put_ieee(img1,img2): #img2 es el logo    
     # I want to put logo on top-left corner, So I create a ROI
     rows,cols,channels = img2.shape
     roi = img1[0:rows, 0:cols ]
     # Now create a mask of logo and create its inverse mask also
     img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
     ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
     mask_inv = cv2.bitwise_not(mask)
     # Now black-out the area of logo in ROI
     img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
     # Take only region of logo from logo image.
     img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
     # Put logo in ROI and modify the main image
     dst = cv2.add(img1_bg,img2_fg)
     img1[0:rows, 0:cols ] = dst
         
     return img1

def put_moustache(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1



    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fc[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fc

def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    hat_width = face_width+1
    hat_height = int(0.35*face_height)+1
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
    return fc

def put_flor(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width)+1
    mst_height = int(face_height*0.3)+1



    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(mst_height):
        for j in range(mst_width):
            for k in range(3):
                if mst[i][j][k]<235:
                    fc[y+i-int(0.25*face_height)][x+j][k] = mst[i][j][k]
    return fc

def put_dog_filter(dog,fc,x,y,w,h):
    face_width = w
    face_height = h
    
    dog = cv2.resize(dog,(int(face_width*1.5),int(face_height*1.75)))
    for i in range(int(face_height*1.75)):
        for j in range(int(face_width*1.5)):
            for k in range(3):
                if dog[i][j][k]<235:
                    fc[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = dog[i][j][k]
    return fc

ch = 0
print("Selecciona un filtro: 1.) Sombrero 2.) Bigote 3.)Sombrero y bigote 4.)Perro 5.)flor ")
ch = int(input())
    
    
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    if ch<1 or ch>5:
        break

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame,"Person Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        if ch==2:
            frame = put_moustache(mst,frame,x,y,w,h)
        elif ch==1:
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==3:
            frame = put_moustache(mst,frame,x,y,w,h)
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==4:
            frame = put_dog_filter(dog,frame,x,y,w,h)
        elif ch==5:
            frame = put_flor(flor,frame,x,y,w,h)
        else:
            break
            
            
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Si quieres salir pulsa q para tweetear la imagen pulsa t', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()