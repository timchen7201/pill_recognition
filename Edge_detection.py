import numpy as np
import argparse
import cv2
import math
from math import ceil
import os
import glob

#

def displayImg(img,windowName):
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName,1000,800)
    cv2.imshow(windowName,img)
    cv2.waitKey(5000)
    # cv2.destroyAllWindows()
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

def getPicture(cnts,image,resized):
    h_avg_list=[]
    w_avg_list=[]
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        h_avg_list.append(math.ceil(h*image.shape[0]/450))
        w_avg_list.append(math.ceil(w*(image.shape[1]/resized.shape[1])))
    h_avg=sum(h_avg_list)/len(h_avg_list)
    w_avg=sum(w_avg_list)/len(w_avg_list)

    for (i, c) in enumerate(cnts):

        (x, y, w, h) = cv2.boundingRect(c)
        # print("->.{0}".format(image.shape[0]*450/image.shape[1]))
        # coin = resized[y:y + h, x:x + w]
        y=math.ceil(y*image.shape[0]/450)
        h=math.ceil(h*image.shape[0]/450)
        x=math.ceil(x*(image.shape[1]/resized.shape[1]))
        w=math.ceil(w*(image.shape[1]/resized.shape[1]))
        print("i:{}\t h:{}\tw:{}".format(i,h,w))
        coin=image[ceil(y*0.9):ceil(y + h*1.1), ceil(x*0.9):ceil(x + w*1.1)]
        # cv2.imshow("Coin", coin)
        cv2.imwrite("./test/output{}.jpg".format(str(i)),coin)
        if(h>100 and h<1000) and ( w>100 and w <1000):
            cv2.imwrite("./pill_reciever_splited/output{}.jpg".format(str(i)),coin)

def splite_image(filename):
    #預先刪除資料夾的照片確保資料夾內是空的
    files = glob.glob("./pill_reciever_splited/*")
    for f in files:
        os.remove(f)
    #載入圖片
    image=cv2.imread(str("./pill_reciever_unsplited/"+filename))
    #resize
    r = 450.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 450)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #高斯
    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    img=blurred
    #Canny 運算子
    edged = cv2.Canny(img, 0, 150)
    #尋找輪廓
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("I count {} pills in this image".format(len(cnts)))
    print(cnts[0].shape)## the coordinate of contours
    coins = resized.copy()
    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
    # displayImg(coins, "Coins")
    getPicture(cnts,image,resized)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    print(args.f)
    image=cv2.imread(str("./pill_reciever_unsplited/"+args.f))
    r = 450.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 450)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # img=resized
    #高斯
    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # displayImg(blurred,"blurred")
    img=blurred
    edged = cv2.Canny(img, 0, 150)
    # displayImg(edged, "Edged")
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("I count {} coins in this image".format(len(cnts)))
    print(cnts[0].shape)## the coordinate of contours
    coins = resized.copy()
    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
#    displayImg(coins, "Coins")
    getPicture(cnts,image,resized)



