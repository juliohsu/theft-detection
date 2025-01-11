import cv2 as cv
import numpy as np

# list of hilook rtsp links
h_rtsp = [
    f"rtsp://admin:Admin123@192.168.80.213:554/Streaming/Channels/{i}02"
    for i in range(1, 17)
]

# list of video captures
caps = [cv.VideoCapture(url) for url in h_rtsp]

# set the cv window full screen
cv.namedWindow("frames", cv.WINDOW_NORMAL)
cv.setWindowProperty("frames", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


# create frames grid
def create_frames_grid(frames, grid_size):
    # resize frames
    frame_size = (160, 120)
    resized_frames = [
        (
            cv.resize(frame, frame_size)
            if frame is not None
            else np.zeros((120, 160, 3), dtype=np.uint8)
        )
        for frame in frames
    ]
    # add blank frames if its necessary
    blank_frame = np.zeros_like(resized_frames[0])
    padded_frames = resized_frames + [blank_frame] * (
        grid_size[0] * grid_size[1] - len(resized_frames)
    )
    # stack frames into grid
    rows = [
        np.hstack(padded_frames[i * grid_size[1] : (i + 1) * grid_size[1]])
        for i in range(grid_size[0])
    ]
    return np.vstack(rows)


while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    frames_grid = create_frames_grid(frames, (4, 4))
    cv.imshow("frames", frames_grid)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps:
    cap.release()
cv.destroyAllWindows()
