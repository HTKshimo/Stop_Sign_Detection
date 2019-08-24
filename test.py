stopsign_cascade = cv2.CascadeClassifier('stopsign-cascade-10stages.xml')
cap = cv2.VideoCapture(0)
# current_image_path = '0001_0050_0055_0027_0027.jpg'
while 1:
    img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stopsigns = stopsign_cascade.detectMultiScale(gray, 50, 50)
    for (x,y,w,h) in stopsigns:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
