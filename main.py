import cv2

# CAP_DSHOW = backend yang lebih cepat & stabil di Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Hand Box Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # tekan 'q' buat keluar
        break

cap.release()
cv2.destroyAllWindows()