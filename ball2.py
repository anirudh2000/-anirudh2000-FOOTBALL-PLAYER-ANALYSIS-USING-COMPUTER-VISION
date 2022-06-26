import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

options = {"model": "yolo_custom.cfg",
           "load": 2300,
           "gpu": 0}

tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()

ref3=cv2.imread('ref1.jpg')


import pprint as pp
from image import dehaze
from tkinter.filedialog import askopenfilename

def ball_detection(path):
    # path=askopenfilename()
    dehaze(path)
    path1='J.png'

    original_img = cv2.imread(path1)
    

    # kernel = np.array([[0, -1, 0],
    # [-1, 5,-1],
    # [0, -1, 0]])
    # original_img = cv2.filter2D(src=original_img, ddepth=-1, kernel=kernel)

    original_img=cv2.resize(original_img,(ref3.shape[0],ref3.shape[1]))
    #original_img=original_img[:int(ref3.shape[0]*.75),:]

    img_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    original_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet2.return_predict(original_img)
    print(results)

    def boxing(original_img,predictions):
        cord=None
        newImage = np.copy(original_img)
        for result in predictions:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']
            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))
            
            if confidence > .13:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
                cord=(top_x, top_y,btm_x, btm_y)  
        return newImage,cord


    img,cord=boxing(original_img, results)    
    print(type(img))

    cv2.imshow('out',img)
    cv2.waitKey(0)
    return cord