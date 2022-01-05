import cv2
import cvlib as cv

vid = cv2.VideoCapture(0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# Loading "Mouths.xml" to train the clasiifier
mouth_cascade = cv2.CascadeClassifier('Mouths.xml')
bw_threshold = 80

while (1):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    # Converts the image from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, bw) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    faces, confidences = cv.detect_face(frame)
    # Detects the face in the image
    if (len(faces) == 0):
        cv2.putText(frame, 'NO FACE DETECTED', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (w, h), (0, 128, 0), 2)
        mask = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        if (len(mask) == 0):
            cv2.putText(frame, 'THANK YOU FOR WEARING MASK', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mask:
                if(y < my < y+h):
                    cv2.putText(frame, 'NOT WEARING MASK', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('image', frame)
    # Press c on keyboard to exit
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break

cv2.destroyAllWindows()
vid.release()
