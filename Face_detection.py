import cv2
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("ArjunPython1307/Hello-AI/haarcascade_frontalface_default.xml")
while True:
    ret,frame = video.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(grey,scaleFactor = 1.1,minNeighbors = 5,minSize = (30,30))
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Face Detection",frame)
video.release()
cv2.destroyAllWindows()