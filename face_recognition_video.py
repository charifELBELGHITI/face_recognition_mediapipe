import cv2
import mediapipe as mp

cap = cv2.VideoCapture("C:/Users/user/Downloads/Video/WhatsApp Video 2023-10-27 at 20.24.22_7285c426.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while cap.isOpened():
    res, img = cap.read()
    if not res:
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

    cv2.imshow('face recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
