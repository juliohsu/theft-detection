import cv2 as cv
from ultralytics import YOLO

# load model
yolo = YOLO(
    ".../best.pt"
)

# load video capture (7 back / 16 top)
videoCap = cv.VideoCapture(0)

# hilook url
# rtsp://admin:Admin123@192.168.80.XXX:554/Streaming/Channels/1602

# intelbras url
# rtsp://admin:Admin123@192.168.80.XXX:554/cam/realmonitor?channel=1&subtype=0

# increase streaming quality
videoCap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
videoCap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
videoCap.set(cv.CAP_PROP_FPS, 30)
videoCap.set(cv.CAP_PROP_BUFFERSIZE, 1)

# name the cv window and set it to fullscreen
cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# get class colour
def getColours(cls_num):
    base_colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colours)
    increments = [(1, -2, 1), (-2, 1, 1), (1, -1, 2)]
    color = [
        base_colours[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colours)) % 256
        for i in range(3)
    ]
    return tuple(color)


while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    for result in results:
        # get classes names
        classes_names = result.names
        # iterate over each box
        for box in result.boxes:
            # check if the box is over 40% confidence
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert coordinates int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # get class id
                cls_num = int(box.cls[0])
                # get class name
                cls_name = classes_names[cls_num]
                # get class colour
                cls_colour = getColours(cls_num)
                # draw a rectangle for
                cv.rectangle(frame, (x1, y1), (x2, y2), cls_colour, 2)
                # write a description text
                cv.putText(
                    frame,
                    f"{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}",
                    (x1, y1),
                    cv.FONT_HERSHEY_COMPLEX,
                    1,
                    cls_colour,
                    2,
                )
    # show the image
    cv.imshow("frame", frame)
    # break the loop if "q" is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# release video capture and destroy all the windows
videoCap.release()
cv.destroyAllWindows()
