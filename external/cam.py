import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


rval, frame = vc.read()

while True:

  if frame is not None:   
     prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     cv2.imshow("preview", prevgray)
  rval, frame = vc.read()

  if cv2.waitKey(1) & 0xFF == ord('q'):
     break


