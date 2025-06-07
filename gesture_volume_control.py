import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess

# Set system volume using amixer (Linux only)
def set_volume_linux(percent):
    subprocess.call(["amixer", "sset", "Master", f"{int(percent)}%"])

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]   # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]   # Index tip
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            length = math.hypot(x2 - x1, y2 - y1)
            volPerc = np.interp(length, [20, 150], [0, 100])
            set_volume_linux(volPerc)

            # Draw visuals
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
            volBar = np.interp(length, [20, 150], [400, 150])
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPerc)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 2)

    cv2.imshow("Gesture Volume Control [Linux]", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
