import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


ref1=cv2.imread('ref1.jpg')
ref2=np.zeros((ref1.shape[0],ref1.shape[1],3))

radius1 = 10
color = (255, 255, 255)
thickness =-1

radius = 3
color = (255, 255, 255)
thickness =-1

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org

  
# fontScale
fontScale = .5
   
# Blue color in BGR
color1 = (255, 255, 255)
  
# Line thickness of 2 px
font_thickness = 2

ref5=cv2.imread('ref1.jpg')

h,w=ref5.shape[:2]

tl=(60,200)
bl=(60,w-190)
lm=(60,w-300)
br=(w-60,w-190)
tr=(w-60,200)
rm=(w-60,w-300)
bm1=(230,w-190)
bm2=(390,w-190)
tm1=(230,200)
tm2=(390,200)



from ball import ball_detection



def check(x,y):
    if x>tl[0] and x<=tm1[0] and y>tl[1] and y<=lm[1]:
        return 'c00'
    elif x>tm1[0] and x<=tm2[0] and y>tl[1] and y<=lm[1]:
        return 'c01'
    elif x>tm2[0] and x<tr[0] and y>tl[1] and y<=lm[1]:
        return 'c02'
    elif x>tl[0] and x<=tm1[0] and y>lm[1] and y<=bl[1]:
        return 'c10'
    elif x>tm1[0] and x<=tm2[0] and y>lm[1] and y<=bl[1]:
        return 'c11'    
    else:
        return 'c12'


def run(path):
    ref4=cv2.imread('ref1.jpg')
    ref1=np.zeros((ref4.shape[0],ref4.shape[1],3))
    h,w=ref1.shape[:2]
   
    #top -left
    ref1 = cv2.circle(ref1,tl, radius, color, thickness)
    #bottom-left
    ref1 = cv2.circle(ref1,bl, radius, color, thickness)
    #left-middle
    ref1 = cv2.circle(ref1,lm, radius, color, thickness)
    #bottom-right
    ref1 = cv2.circle(ref1,br, radius, color, thickness)
    #right -middle
    ref1 = cv2.circle(ref1,rm, radius, color, thickness)
    #top-right
    ref1 = cv2.circle(ref1,tr, radius, color, thickness)
    #bottom-middle1
    ref1 = cv2.circle(ref1,bm1, radius, color, thickness)
    #bottom-middle2
    ref1 = cv2.circle(ref1,bm2, radius, color, thickness)

    #top-middle1
    ref1 = cv2.circle(ref1,tm1, radius, color, thickness)
    #top-middle2
    ref1 = cv2.circle(ref1,tm2, radius, color, thickness)

    #lines

    line_thickness = 2
    cv2.line(ref1, tl, tr, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, bl, br, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, tl, bl, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, tr, br, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, tm1, bm1, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, tm2, bm2, (0, 255, 0), thickness=line_thickness)
    cv2.line(ref1, lm, rm, (0, 255, 0), thickness=line_thickness)


    
    total=0
    c00=0
    c01=0
    c02=0
    c10=0
    c11=0
    c12=0

    ref2=np.zeros((ref4.shape[0],ref1.shape[1],3))
    paths=os.listdir(path) 
    print(paths,'======')
    for path1 in paths:
        try:
            topx,topy,botx,boty=ball_detection(path+'/'+path1)
            ctnx=topx+(botx-topx)//2
            ctny=topy+(boty-topy)//2
            center_coordinates = (ctnx, ctny+100)
            ret=check(ctnx, ctny+100)
            if ret=='c00':
                c00+=1
            elif ret=='c01':
                c01+=1 
            elif ret=='c02':
                c02+=1 
            elif ret=='c10':
                c10+=1 
            elif ret=='c11':
                c11+=1 
            elif ret=='c12':
                c12+=1  
            total+=1                      
            print(center_coordinates)
            print(ref2.shape)
            
            ref2 = cv2.circle(ref2, center_coordinates, radius1, color1, -1)
           
        except Exception as e:
            print(e)
            pass   

    #========

    c00p=round((c00/total)*100,2)
    c01p=round((c01/total)*100,2)
    c02p=round((c02/total)*100,2)
    c10p=round((c10/total)*100,2)
    c11p=round((c11/total)*100,2)
    c12p=round((c12/total)*100,2)
    
    org = (tl[0]+50, tl[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c00p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)

    org = (tm1[0]+50, tl[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c01p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)

    org = (tm2[0]+50, tl[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c02p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)

    org = (tl[0]+50, lm[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c10p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)

    org = (bm1[0]+50, lm[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c11p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)

    org = (bm2[0]+50, lm[1]+50)
    ref1= cv2.putText(ref1, '%s'%(c12p), org, font, fontScale, color, font_thickness, cv2.LINE_AA)



    #========        
    cv2.imwrite('ref2.jpg',ref2) 
    ref2=cv2.imread('ref2.jpg')
    lab = cv2.cvtColor(ref2, cv2.COLOR_BGR2LAB)
    a_component = lab[:,:,1]
    th = cv2.threshold(a_component,140,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite('th.jpg',th) 
    blur = cv2.GaussianBlur(ref2,(13,13), 11)
    cv2.imwrite('blur.jpg',blur) 
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    cv2.imwrite('heat.jpg',heatmap_img) 
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, ref4, 0.5, 0)
    cv2.imwrite('ref3.jpg',super_imposed_img)
    cv2.imwrite('percentage.jpg',ref1)    





if __name__=="__main__":
    path='main'
    run(path)