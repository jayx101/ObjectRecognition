import cv2

print cv2.__version__

cap = cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2(200, -1, True)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('frame1', fgmask)

    # cv2.imwrite("blob.jpg", fgmask) # writes image test.bmp to disk

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
