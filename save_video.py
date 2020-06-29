import os
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

path = 'Face_Library'
images = []
classNames = []   # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = [f for f in os.listdir(path) if not f.startswith('.')]
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        dir = os.path.join(path,cl)
        imgList = [f for f in os.listdir(dir) if not f.startswith('.')]
        imgPath = os.path.join(dir,imgList[-1])
        curImg = cv2.imread(imgPath)
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

fontpath = "utils/simsun.ttc" # <== 这里是宋体路径 
font = ImageFont.truetype(fontpath, 32)

save_path ='Results'
os.makedirs(save_path, exist_ok=True)

encodeListKnown = findEncodings(images)
print('Encodings Complete')

video_path = 'Demo_Video/SistersWhoMakeWaves_EP2_Short.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Cannot capture source'
print("Start")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
file_name = os.path.join(save_path,os.path.basename(video_path))
out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

scale = 1
while cap.isOpened():
    success, img = cap.read()
    if success:
        imgS = cv2.resize(img,(0,0),None,1/scale,1/scale)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
    
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*scale,x2*scale,y2*scale,x1*scale
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.rectangle(img,(x1,y2+32),(x2,y2),(255,0,0),cv2.FILLED)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1,y2),  name, font = font, fill = (255,255,255,0))
                img = np.array(img_pil)
                # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
        out.write(img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        continue
    else:
        break

cap.release()
out.release()



