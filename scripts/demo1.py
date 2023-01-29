import cv2
cap=cv2.VideoCapture(0)
while 1:
    ret,frame=cap.read()
    cv2.imshow('video',frame)
    cv2.waitKey(1)