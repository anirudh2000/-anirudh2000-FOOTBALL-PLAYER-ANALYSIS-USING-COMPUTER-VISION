import tkinter as tk
from PIL import ImageTk, Image
import cv2
import numpy as np
#from detection1 import *
from tkinter.filedialog import askdirectory
from process import run

window = tk.Tk()
window.title("Goal keeper analysis")
window.configure(background='Black')


def Detect():
    dir1=askdirectory()
    print(dir1)
    run(dir1)
  
    
    # Read First Image
    img1 = cv2.imread('ref3.jpg')
    
    # Read Second Image
    img2 = cv2.imread('percentage.jpg')
    
    
    
   # concatenate image Horizontally
    Hori = np.concatenate((img1, img2), axis=1)
    
   
    
    cv2.imshow('Result', Hori)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "football.jpg"
img=Image.open(path)
img=img.resize((600,300))

img = ImageTk.PhotoImage(img)

panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

b2=tk.Button(panel,text="Choose folder", command=Detect)
b2.pack(side="left")
b2.place(x=250,y=150)

window.mainloop()