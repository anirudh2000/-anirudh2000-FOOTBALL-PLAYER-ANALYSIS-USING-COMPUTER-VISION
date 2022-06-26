import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename



def white(path)

    img = cv2.imread(path)   # you can read in images with opencv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sensitivity = 110
    hsv_color1 = np.array([0,0,255-sensitivity])
    hsv_color2 = np.array([255,sensitivity,255])



    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

    plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
    plt.show()


if __name__=="__main__":
    path=askopenfilename()
    white(path)

