
# coding: utf-8

# In[ ]:


import cv2

face_cascade_path = 'D:\jupyter\package\opencv\data\haarcascades\haarcascade_frontalface_default.xml'
eye_cascade_path = 'D:\jupyter\package\opencv\data\haarcascades\haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

imwww = cv2.imread('D:\jupyter\Image_Processing\warai1.jpg')
src = cv2.imread('D:\jupyter\Image_Processing\hogepiyo.jpg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(src_gray)

for x, y, w, h in faces:
    img2 = cv2.resize(imwww, (w, h))
    roi = src[y: y + h, x: x + w ]
    
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    
    dst = cv2.add(img1_bg,img2_fg)
    src[y: y + h, x: x + w ] = dst

cv2.imwrite('D:\jupyter\Image_Processing\opencv_face_detect_rectangle.jpg', src)
# True

