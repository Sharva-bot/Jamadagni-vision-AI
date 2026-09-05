import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (9,10),(10,11),(11,12),        # Middle
    (13,14),(14,15),(15,16),      # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17)            # Palm base
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.5)
    
    annotated = results[0].plot(kpt_line=False, kpt_radius=0)
    
    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()
        for hand in kpts:
            # Draw green connecting lines
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(hand) and p2 < len(hand):
                    x1, y1 = int(hand[p1][0]), int(hand[p1][1])
                    x2, y2 = int(hand[p2][0]), int(hand[p2][1])
                    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw green joint dots
            for x, y in hand:
                if x > 0 and y > 0:
                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    cv2.imshow("Janadagni 15 Live Test", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
