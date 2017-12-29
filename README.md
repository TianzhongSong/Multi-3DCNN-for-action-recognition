# Multi-3DCNN-for-action-recognition
A simple multi-3DCNN framework for action recognition using global and local information

### requirement
python2.7 opencv3.0 keras2.0.8 tensorflow1.4 

### files
#### avi2img.py (make dataset)
Use SSD (https://github.com/rykov8/ssd_keras) to detect people and save images.

KTH dataset:http://www.nada.kth.se/cvap/actions/

![original and crop images](https://github.com/TianzhongSong/Multi-3DCNN-for-action-recognition/blob/master/images.png)

#### model.py
define the models

#### train.py
train the 3DCNN models with global and local information respectively.

#### evaluate.py
extract 3dcnn feature with pretrained models and evaluate features with svm.

## framework
![framework](https://github.com/TianzhongSong/Multi-3DCNN-for-action-recognition/blob/master/framework.png)

see details in model.py, train.py and evaluate.py scripts.

## results

### global 
only use global information (total acc:81.40%)
![global confusion matrix](https://github.com/TianzhongSong/Multi-3DCNN-for-action-recognition/blob/master/global_confusion_matrix.jpg)

### crop 
only use local information (total acc:79.07%)
![crop confusion matrix](https://github.com/TianzhongSong/Multi-3DCNN-for-action-recognition/blob/master/crop_confusion_matrix.jpg)

### merge 
merge global and local information (total acc:87.44%)
![merge confusion matrix](https://github.com/TianzhongSong/Multi-3DCNN-for-action-recognition/blob/master/merge_confusion_matrix.jpg)
