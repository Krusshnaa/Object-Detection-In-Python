# Object Detection using SSD MobileNet V3 in Python

This project demonstrates real-time object detection using the **SSD MobileNet V3** architecture, with pre-trained weights from the **COCO dataset**. The model is used to detect objects in a webcam feed using OpenCV.

## What is MobileNet?

**MobileNet** is a family of neural network architectures optimized for mobile and embedded vision applications. **SSD (Single Shot MultiBox Detector)** combined with **MobileNet V3** is lightweight and designed for efficient object detection in real-time on devices with limited computational resources.

## Files Used

- **ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt**: Configuration file specifying the model's architecture and layers.
- **frozen_inference_graph.pb**: Pre-trained model weights, frozen for inference.

## Requirements

- OpenCV (`opencv-python`)
- Python 3.x
- Webcam or video input device

## Installation

1. Clone the repository or download the project files.
2. Install the required libraries:
    ```bash
    pip install opencv-python
    ```

3. Download the necessary model files:
   - [ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt](ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)
   - [frozen_inference_graph.pb](frozen_inference_graph.pb)

## How to Run

1. Ensure that your webcam is connected.
2. Run the `object_detection.py` script:
   ```bash
   python object_detection.py
   ```
   The webcam will open, and you will see real-time detection of objects with bounding boxes and labels.

## Code Snippets

### Importing Libraries and Initial Setup

```python
import cv2

def Camera():
    cam = cv2.VideoCapture(1)  # Capture from Webcam 
    cam.set(3, 740)  # Set width
    cam.set(4, 580)  # Set height
```
##Loading Class Names and Model
```bash
  

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath, configPath)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
```
##Object Detection and Display
``` bash
while True:
    success, img = cam.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
```
##Saving Video Output
``` bash
while True:
    success, img = cam.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
```
##Exiting the Application
``` Bash
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
out.release()
cv2.destroyAllWindows()
```
##Object Classes
This model is trained on the COCO dataset and can detect the following object categories:
```bash
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```
Here's the `README.md` section formatted to include the complete workflow and additional information:


## Complete Workflow

1. **Connect your Webcam**: Ensure that the webcam is properly connected and functioning.
2. **Run the Script**: Run the object detection script `object_detection.py`. The video feed from your webcam will be displayed with bounding boxes around detected objects.
3. **Press 'q' to Exit**: You can press the `q` key anytime to exit the application and stop the video feed.
4. **Save Video Output**: The detected output is saved in the file `output.avi` by default.

## Additional Information

- This project is designed for lightweight applications, suitable for real-time object detection on resource-constrained devices.
- The **SSD MobileNet V3** model used here is trained on the **COCO dataset**, which can recognize up to 90 different objects in real-time.
- Make sure to adjust the webcam source (e.g., `cv2.VideoCapture(1)`) depending on your webcam setup. It might be necessary to use `cv2.VideoCapture(0)` or another index if the default one doesn't work.


