# theft-detection

Firstly this is a self implement yolo theft detection just for fun <3

And second i added a simple theft detection tutorial by using models like YOLOV11 or YOLO8S (OBS: the difference between this two, the first one have a segmentation and the second one doesnt have, for more information please check out ultralytics website). And then for model training/inference, with ROBOFLOW to explore/download a universe of cash theft and shoplifting universe, etc.

- "main.py" is a file that theft dedicated detection by combining ULTRALYTICS and OPENCV, to output the theft detection on your device camera if you have interest you can output your camera streaming directly or local src file, with a trained yolo model from the model.ipynb.

- "src" is a folder that contains images and videos for testing purpose, if you want to use it for training i recommend to use labelimg for image and videos frame labeling, then output those into a separate files like "train, valid, test" to train your model.

- "models" is a folder where initially has a yolo model tutorial for you to train your own yolo model, but if you want you can choose another models from tensorflow/pytorch etc.

- "hilook-cameras" is a file that shows you how to retrieve cameras videos by its rtsp url and output with the opencv library.

- "yolo-model-detect" is a file where you get more options for retrieving cameras videos such as INTELBRAS, or even your notebook.