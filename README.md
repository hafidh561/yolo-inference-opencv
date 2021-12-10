# YOLOv4 Inference OpenCV

## Description

Inference your YOLOv4 model with OpenCV.

## Installation

> **Python version 3.6 or newer** \
> **If you want use cuda for inference, install opencv with cuda support.**

```bash
git clone https://github.com/hafidh561/yolo-inference-opencv.git
pip install opencv-python==4.5.3.56
```

## Usage

```bash
$ python app.py -h
usage: app.py [-h] [-s SOURCE_FILE] [-c CONFIDENCE_THRESHOLD]
              [--show-file SHOW_FILE] [--save-file SAVE_FILE]
              [--weights-path WEIGHTS_PATH] [--config-path CONFIG_PATH]
              [--size-model SIZE_MODEL] [--obj-names OBJ_NAMES]
              [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE_FILE, --source-file SOURCE_FILE
                        Input your file path to detect the objects or input
                        'webcam' to detect the objects with your webcam
  -c CONFIDENCE_THRESHOLD, --confidence-threshold CONFIDENCE_THRESHOLD
                        Input your minimal confidence to detect the objects
  --show-file SHOW_FILE
                        Do you want to show your file with window? True or
                        False
  --save-file SAVE_FILE
                        Do you want to save your file? True or False
  --weights-path WEIGHTS_PATH
                        Input your darknet path weights model
  --config-path CONFIG_PATH
                        Input your darknet path config model
  --size-model SIZE_MODEL
                        Input your darknet size image shape config model
  --obj-names OBJ_NAMES
                        Input your darknet path obj.names
  --device DEVICE       Input your device runtime

```

## Give It a Try

If you want make your own YOLOv4 for object detection? Give it a try in this [Google Colab](https://colab.research.google.com/github/hafidh561/yolo-inference-opencv/blob/main/train_model.ipynb)

## License

[MIT LICENSE](./LICENSE)

© Developed by [hafidh561](https://github.com/hafidh561)
