from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc




pill_classes=[]
database={}


#PillModel = faceRecoModel(input_shape=(3,96,96))
#print("Total Params:", PillModel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
        Implementation of the triplet loss as defined by formula (3)
        Arguments:
        y true -- true labels, required when you define a loss in Keras, you don't need it in
        this function.
        y pred -- python list containing three objects:
        anchor -- the encodings for the anchor images, of shape (None, 128)
        positive -- the encodings for the positive images, of shape (None, 128)
        negative -- the encodings for the negative images, of shape (None, 128)
        Returns:
        loss -- real number, value of the loss
        """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dis=tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dis=tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    
    basic_loss=tf.subtract(pos_dis,neg_dis)+alpha
    
    loss=tf.reduce_sum(tf.maximum(basic_loss,0))
    
    return loss


def read_and_resize(image_path):
    img1=Image.open(image_path)
    img1=scipy.misc.imresize(img1, (2000, 2000))
    img=img1[:,200:2201]
    img=cv2.resize(img,(96,96),interpolation=cv2.INTER_NEAREST)
    plt.imshow(img)
    return img

def creating_database(model):
    for i in os.listdir('images/'):
        if not os.path.isfile(os.path.join('images/',i)):
            count=0
            for pill_name in os.listdir(os.path.join('images/',i)):
                pill_classes.append(str(i+str(count)))
                img_path=os.path.join('images/',i)
                img_path=os.path.join(img_path,pill_name)
                img=read_and_resize(img_path)
                database[str(i+str(count))]=img_to_encoding(img,model)
                count+=1;

def verify_pill(image_path, identity, database, model):
    if type(image_path)==str:
        img1 = cv2.imread(image_path, 1)
    else:
        img1=image_path
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    encoding = model.predict_on_batch(x_train)

    dist=np.linalg.norm(encoding-database[identity])

    return dist

def prepare_model():
    PillModel=PillRecoModel(input_shape=(3,96,96))
    PillModel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
    load_weights_from_FaceNet(PillModel)
    return PillModel


def recognize(img_rec,model):
    scores={}
    score_list=[]
    score_avg={}
    img_to_test=read_and_resize(img_rec)
    for i in pill_classes:
        scores[i]=verify_pill(img_to_test,i,database,model)
        score_list.append(verify_pill(img_to_test,i,database,model))
    k=0
    for i in range(len(database))[::23]:
        if i!=0:
            tmp_list=score_list[k:i]
            avg=sum(tmp_list)/len(tmp_list)
            score_avg[list(scores.keys())[k+1]]=avg
            k=i
    reco_pill=min(score_avg, key=lambda k: score_avg[k])
    return reco_pill


if __name__=='__main__':
    creating_database()
    #print(pill_classes)
    model=prepare_model()
    print("Total Params:", PillModel.count_params())

