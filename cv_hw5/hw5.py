import cv2
import numpy as np
kernel = [[0,10,10,10 ,0],
          [10,10,10,10,10],
          [10,10,10,10,10],
          [10,10,10,10,10],
          [0,10,10,10,0]]
def expand (img,size):
    expand = []
    img_size=len(img)
    for i in range (size):
        expand.append([])
        for j in range (size*2+img_size):
            expand[i].append(0)
    for i in range (img_size):
        expand.append([])
        for j in range (size):
            expand[size+i].append(0)
        for j in range (img_size):
            expand[size+i].append(img[i][j][0])
        for j in range (size):
            expand[size+i].append(0)
    for i in range (size):
        expand.append([])
        for j in range (size*2+img_size):
            expand[i+img_size+size].append(0)
    return expand

def dilation (img):
    img_size = len(img)
    size =2
    exp=expand(img,2)
    for i in range(size,img_size+size):
        for j in range (size,img_size+2):
            if(exp[i][j]!=0):
                iter =0
                for ii in range (i-2,i+3):
                    for jj in range (j-2,j+3):
                       if(iter!=0 and iter !=4 and iter!=20 and iter!=24 ):
                           exp[ii][jj]+=5
                       iter+=1 



    for i in range (len(exp)):
        for j in range (len(exp)):
            exp[i][j]=[exp[i][j],exp[i][j],exp[i][j]]
    a =np.array(exp)   
    cv2.imwrite("dilation_gs.jpg",a)

def main ():
    img = cv2.imread("lena.bmp")
    dilation(img)
    
    #cv2.imwrite("asfd.jpg",img)
    



if __name__ == '__main__':
    main()