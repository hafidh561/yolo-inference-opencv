import cv2
import argparse
import mimetypes
from yolo_opencv import YoloOpenCV


def boolean_string(s):
    if s == "0":
        s = "False"
    elif s == "1":
        s = "True"
    else:
        s = s.title()
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string!")
    return s == "True"


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source-file",
    default="./source/image.jpg",
    help="Input your file path to detect the objects or input 'webcam' to detect the objects with your webcam",
)
parser.add_argument(
    "-c",
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Input your minimal confidence to detect the objects",
)
parser.add_argument(
    "--show-file",
    type=boolean_string,
    default=False,
    help="Do you want to show your file with window? True or False",
)
parser.add_argument(
    "--save-file",
    type=boolean_string,
    default=True,
    help="Do you want to save your file? True or False",
)
parser.add_argument(
    "--weights-path",
    default="./models/coco-dataset/yolov4.weights",
    help="Input your darknet path weights model",
)
parser.add_argument(
    "--config-path",
    default="./models/coco-dataset/yolov4.cfg",
    help="Input your darknet path config model",
)
parser.add_argument(
    "--size-model",
    type=int,
    default=608,
    help="Input your darknet size image shape config model",
)
parser.add_argument(
    "--obj-names",
    default="./models/coco-dataset/obj.names",
    help="Input your darknet path obj.names",
)
parser.add_argument(
    "--device",
    default="cpu",
    help="Input your device runtime",
)
value_parser = parser.parse_args()
source_file = value_parser.source_file
show_file = value_parser.show_file
save_file = value_parser.save_file
conf_thresh = value_parser.confidence_threshold
device = value_parser.device
weights = value_parser.weights_path
config = value_parser.config_path
obj_names = value_parser.obj_names
size_model = value_parser.size_model
real_name_file = source_file.split("/")[-1]
detect_object_file = None

OUTPUT_SIZE_WIDTH = 640
OUTPUT_SIZE_HEIGHT = 480

yolo = YoloOpenCV(
    weights=weights,
    config=config,
    obj_names=obj_names,
    size_model=size_model,
    device=device,
    conf_thresh=conf_thresh,
)

if source_file != "webcam":
    mimestart = mimetypes.guess_type(source_file)[0]
    if mimestart != None:
        mimestart = mimestart.split("/")[0]
        if mimestart in ["image"]:
            detect_object_file = "image"
        elif mimestart in ["video"]:
            detect_object_file = "video"
        else:
            raise ValueError("Input your source file correctly!")
elif source_file == "webcam":
    detect_object_file = "webcam"

if detect_object_file == "image":
    img = cv2.imread(source_file)
    img = cv2.resize(img, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
    print("Predict process....")
    img = yolo.predict(img)
    if save_file and show_file:
        print("Predict success!")
        cv2.imwrite(f"./results/{real_name_file}", img)
        print(f"Your file has been saved in /results/{real_name_file}")
        while True:
            cv2.imshow(real_name_file, img)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
    elif save_file and not show_file:
        print("Predict success!")
        cv2.imwrite(f"./results/{real_name_file}", img)
        print(f"Your file has been saved in /results/{real_name_file}")
    elif not save_file and show_file:
        print("Predict success!")
        while True:
            cv2.imshow(real_name_file, img)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        print("Predict success!")
        exit()

elif detect_object_file == "video":
    cap = cv2.VideoCapture(source_file)
    print("Predict process....")
    if save_file and show_file:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            f"./results/{real_name_file.split('.')[0]}.avi",
            fourcc,
            20.0,
            (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT),
        )
        print("Predict success!")
        while cap.isOpened():
            result, img = cap.read()
            if not result:
                break
            img = cv2.resize(img, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
            img = yolo.predict(img)
            cv2.imshow(real_name_file, img)
            out.write(img)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(
            f"Your file has been saved in /results/{real_name_file.split('.')[0]}.avi"
        )
    elif save_file and not show_file:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            f"./results/{real_name_file.split('.')[0]}.avi",
            fourcc,
            20.0,
            (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT),
        )
        while cap.isOpened():
            result, img = cap.read()
            if not result:
                break
            img = cv2.resize(img, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
            img = yolo.predict(img)
            out.write(img)
        print("Predict success!")
        cap.release()
        out.release()
        print(
            f"Your file has been saved in /results/{real_name_file.split('.')[0]}.avi"
        )
    elif not save_file and show_file:
        print("Predict success!")
        while cap.isOpened():
            result, img = cap.read()
            if not result:
                break
            img = cv2.resize(img, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
            img = yolo.predict(img)
            cv2.imshow(real_name_file, img)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Predict success!")
        exit()

elif detect_object_file == "webcam":
    cap = cv2.VideoCapture(0)
    cap.set(3, OUTPUT_SIZE_WIDTH)
    cap.set(4, OUTPUT_SIZE_HEIGHT)
    print("Predict process....")
    if save_file and not show_file:
        _, img = cap.read()
        img = yolo.predict(img)
        print("Predict success!")
        cv2.imwrite(f"./results/{real_name_file}", img)
        cap.release()
        print(f"Your file has been saved in /results/{real_name_file}")
    elif not save_file and show_file:
        print("Predict success!")
        while cap.isOpened():
            _, img = cap.read()
            img = yolo.predict(img)
            cv2.imshow("Webcam", img)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Predict success!")
        exit()
