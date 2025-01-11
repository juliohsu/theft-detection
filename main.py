from ultralytics import YOLO
import cv2 as cv

model = YOLO("")

cv.namedWindow("Robbery Detection", cv.WINDOW_NORMAL)
cv.setWindowProperty("Robbery Detection", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

cap = cv.VideoCapture("videos/events/normal/receive_and_give.mp4")
if not cap.isOpened():
    exit()


def getColour(id):
    base_colour = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    increments = [(1, -2, 1), (-2, 1, 1), (1, -1, 2)]
    colour_index = id % len(base_colour)
    colour = [
        base_colour[colour_index][i]
        + increments[colour_index][i] * (id // len(base_colour)) % 256
        for i in range(3)
    ]
    return tuple(colour)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)
    for result in results:
        classes_name = result.names
        for box in result.boxes:
            # get class id
            id = int(box.cls[0])
            # get class name
            name = classes_name[id]
            # get class colour
            colour = getColour(id)
            # get box coordinates
            [x1, y1, x2, y2] = map(int, box.xyxy[0])
            # get box confidence
            conf = box.conf[0]
            # get box label
            label = f"{name} {conf:2f}"
            # box rectangle
            cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            # box text
            cv.putText(frame, label, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1, colour, 2)
    cv.imshow("Robbery Detection", frame)
    if cv.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
