# -*- coding: utf-8 -*-
"""人工智慧.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D4IaYo3iVUgnWnIUYl-i9_2LrvRD98Dx
"""

import os
cd = os.chdir
cd('/content')
!rm -rf faster-rcnn.pytorch
!git clone -b pytorch-1.0 https://github.com/jwyang/faster-rcnn.pytorch.git

!sudo apt update
!sudo apt install python3.7

!sudo apt install python3-pip
!sudo apt install python3.7-distutils

!python3.7 -m pip install torch==1.0.1

!sudo apt install python3.7-dev

cd('/content/faster-rcnn.pytorch')
!python3.7 -m pip install -r requirements.txt
cd('lib')
!python3.7 setup.py build develop

cd('..')
!mkdir data
os.chdir('data')
!mkdir pretrained_model
os.chdir('pretrained_model')
# 下载预训练模型res101
!wget https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth
# 下载预训练模型vgg16
!wget https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth

os.chdir('../') #返回上一级目录即data/下
# 下载数据集
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
# 解压缩
!tar xvf VOCtrainval_06-Nov-2007.tar
!tar xvf VOCtest_06-Nov-2007.tar
!tar xvf VOCdevkit_08-Jun-2007.tar
# 建立软连接
!ln -s $VOCdevkit VOCdevkit2007 #注意！如果上面解压缩得到的文件夹名字为"VOCdevdit"，要将其改为“VOCdevdit2007"，否则后面会报错。

cd('/content/faster-rcnn.pytorch/data')
!rm VOCdevkit2007
!mv VOCdevkit VOCdevkit2007

#cd('/content')
#!git clone https://github.com/cocodataset/cocoapi.git
#cd('faster-rcnn.pytorch/lib')
#!mv pycocotools pycocotools.backup
#!cp -r /content/cocoapi/PythonAPI/pycocotools .

cd('/content/faster-rcnn.pytorch/data')
!git clone https://github.com/pdollar/coco.git
cd('coco/PythonAPI')
!sed -i 's/python/python3.7/g' Makefile
!make

!python3.7 -m pip install scipy==1.2.1
!python3.7 -m pip install torch==1.0.1 torchvision==0.2.0
!python3.7 -m pip install pyyaml==5.4.1

cd('/content/faster-rcnn.pytorch')

 !CUDA_VISIBLE_DEVICES=0 python3.7 trainval_net.py \
                    --dataset pascal_voc \
                    --net res101 \
                    --bs 4 \
                    --nw 0 \
                    --lr 0.004 \
                    --lr_decay_step 8 \
                    --epochs 10 \
                    --cuda