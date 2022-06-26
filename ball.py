import cv2
import numpy as np
import math


import os
# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("Project_Extra/yolov4.weights",
                      "Project_Extra/yolov4.cfg")
classes = []
with open("Project_Extra/labels.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

ref3=cv2.imread('ref1.jpg')
def ball_detection(path1):
    original_img = cv2.imread(path1)
    

   

    frame0=cv2.resize(original_img,(ref3.shape[0],ref3.shape[1]))
    height, width, channels = frame0.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame0, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.6:
                # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                x1 = int(center_x + w / 2)
                y1 = int(center_y + h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='sports ball':
                color = colors[i]
                cv2.rectangle(frame0, (x-10, y-10), (x + w+20, y + h+20), color, 2)
                cv2.putText(frame0, label, (x, y-15), font, 1, color, 2)
                cord=(x,y,x+w,y+h)

            print(label)

    # cv2.imshow("Video0", frame0)
    # cv2.waitKey(0)
    return cord





if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    # frame=cv2.imread(path)
    ball_detection(path)