import cv2

cap = cv2.VideoCapture(0)

while True:
    sucess, frame = cap.read()

    if not sucess:
        print('Falha em carregar!')

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):

        cap.release()
        cv2.destroyAllWindows()