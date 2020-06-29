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

def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image

fontpath = "utils/simsun.ttc" # <== 这里是宋体路径 
font = ImageFont.truetype(fontpath, 32)

encodeListKnown = findEncodings(images)
print('Encodings Complete')

save_path ='Results'
os.makedirs(save_path, exist_ok=True)

video_path = 'Demo_Video/Fake_Cry.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), 'Cannot capture source'
print("Start")




frame_height = int(cap.get(4))
frame_width = frame_height * 2
fps = cap.get(cv2.CAP_PROP_FPS)
file_name = os.path.join(save_path,os.path.basename(video_path))
out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


scale = 4
while cap.isOpened():
    success, img = cap.read()
    if success:
        height, width, channels = img.shape
        upper_left = (int(width*3.5/16),0)
        bottom_right = (int(width*12.5/16), int(height))
        img = img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        img1 = img.copy()
        img2 = img.copy()
        imgS = cv2.resize(img,(0,0),None,1/scale,1/scale)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*scale,x2*scale,y2*scale,x1*scale
                name = classNames[matchIndex].upper()
                print(name)
                if name == '黄圣依':
                    face = img2[y1:y2, x1:x2]
                    face = anonymize_face_pixelate(face,blocks=10)
                    img2[y1:y2, x1:x2] = face
                else:
                    img2 = img
                img1 = cv2.rectangle(img1,(x1,y1),(x2,y2),(255,0,0),2)
                img1 = cv2.rectangle(img1,(x1,y2+32),(x2,y2),(255,0,0),cv2.FILLED)
                img_pil = Image.fromarray(img1)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1,y2),  name, font = font, fill = (255,255,255,0))
                img1 = np.array(img_pil)
        final_img = cv2.hconcat([img1,img2])
        out.write(final_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        continue
    else:
        break



