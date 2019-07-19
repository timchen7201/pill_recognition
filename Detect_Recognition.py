from os import listdir
from os.path import isfile,join
import time
from Load_Module import recognize
from Load_Module import prepare_model
from Load_Module import creating_database

import Edge_detection
import os
import cv2

def recognize_each_pic(model):
    list_files=[f for f in listdir("./pill_reciever_splited/") if isfile(join("./pill_reciever_splited/",f)) and f.endswith(".jpg")]
    for i in range(len(list_files)):
        ans=recognize('./pill_reciever_splited/'+list_files[i],model)
        print("{0} : {1}".format(list_files[i],ans))
#    cmpnum=len(origin_files)
#    while True:
#        files=[f for f in listdir("./pill_reciever_splited/") if isfile(join("./pill_reciever_splited/",f)) and f.endswith(".jpg")]
#        if len(files)>cmpnum:
#            cmpnum=len(files)
#            list_dif=[i for i in files if i not in origin_files]
#            origin_files=files
#            ans=recongize('./pill_reciever_splited/'+list_dif[0])

        time.sleep(1)
def check_whole_pic():
    origin_files=[f for f in listdir("./pill_reciever_unsplited/") if isfile(join("./pill_reciever_unsplited/",f)) and f.endswith(".jpg")]
    filename=""
    if origin_files==[]:
        while True:
           
            print("wait")
            files=[f for f in listdir("./pill_reciever_unsplited/") if isfile(join("./pill_reciever_unsplited/",f)) and f.endswith(".jpg")]
            if len(files)>0:
                filename=files[0]
                break
        print("Detect:"+filename)
        time.sleep(2)
        Edge_detection.splite_image(filename)
        time.sleep(2)
        os.remove("./pill_reciever_unsplited/"+filename)

        return True



if __name__=="__main__":
    # while True:
    model=prepare_model()
    creating_database(model)
    if check_whole_pic():
        print("recognize")
        recognize_each_pic(model)
