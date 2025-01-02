from ultralytics import YOLO
import cv2 as cv

# fullscreen the video/image output, if you dont want the fullscreen just comment out both of two lines below
cv.namedWindow("Theft Detection", cv.WINDOW_NORMAL)
cv.setWindowProperty("Theft Detection", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# choose model library
model = YOLO(
    ".../best.pt"  # put your trained yolo model from the model.ipynb file url path
)

# source for your model to detect, that it can be camera streaming like the url below, or images/videos that is in the src file
# "rtsp://admin:Admin123@{CAMERA_HDVR_IP}:{CAMERA_PORT}/Streaming/Channels/{CHANNEL_NUM}"
videoCap = cv.VideoCapture(0)

if not videoCap.isOpened():
    exit()


# colour picking function
def getColour(cls_id):
    base_colour = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    colour_index = cls_id % len(base_colour)
    increments = [(1, -2, 1), (-2, 1, 1), (1, -1, 2)]
    colour = [
        base_colour[colour_index][i]
        + increments[colour_index][i] * (cls_id // len(base_colour)) % 256
        for i in range(3)
    ]
    return tuple(colour)


while True:
    ret, frame = videoCap.read()
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

            # get coordinates and convert into int
            [x1, y1, x2, y2] = map(int, box.xyxy[0])
            # box confidence
            conf = box.conf[0]
            # box label
            label = f"{model.names[id]} {conf:2f}"

            # draw box border
            cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            # label the box
            cv.putText(frame, label, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1, colour, 2)

    cv.imshow("Theft Detection", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

videoCap.release()
cv.destroyAllWindows()
