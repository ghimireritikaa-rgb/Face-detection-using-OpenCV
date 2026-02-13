import cv2 as cv

img = cv.imread(r"download (18).jpeg")
resized = cv.resize(img, (500, 500))
gray = cv.cvtColor(resized,  cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

for (x, y, w, h) in faces :
    cv.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow("Faces", resized)
cv.waitKey(0)