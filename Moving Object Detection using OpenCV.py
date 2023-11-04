import cv2  # image
import time  # delay
import imutils  # resize

cam = cv2.VideoCapture(0)  # cam id
time.sleep(0.5)  # time delay for camera initialization

firstFrame = None
area = 500

while True:
    ret, frame = cam.read()  # read frame from camera
    text = "Normal"
    frame = imutils.resize(frame, width=1000)  # resize the resolution

    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # color to grayscale
    grayFrame = cv2.GaussianBlur(grayImg, (21, 21), 0) #smoothened

    if firstFrame is None:
        firstFrame = grayFrame  # capturing 1st frame on the first iteration
        continue

    frameDiff = cv2.absdiff(firstFrame, grayFrame)  # absolute diff between 1st and current frame

    threshFrame = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]  # binary

    threshFrame = cv2.dilate(threshFrame, None, iterations=2)

    cnts = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
    print(text)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

