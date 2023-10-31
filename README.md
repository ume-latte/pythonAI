# pythonAI
南華大學跨領域-人工智慧期中報告
## 組員

* 11125008賴宣佑
* 11125042程莉芸
* 11128009黃祐真

## 作業流程 
* 準備一個可以使用google colab 的帳號
* 下載faster r-cnn的實作成品
``` python
import os
cd = os.chdir
cd('/content')
!rm -rf faster-rcnn.pytorch
!git clone -b pytorch-1.0 https://github.com/jwyang/faster-rcnn.pytorch.git
```
* 更新機器並下載需使用之資料
``` python
!sudo apt update
!sudo apt install python3.7
```
* 下載安裝distutils模組，然後使用Python 3.7的pip工具安裝特定版本的Torch庫（1.0.1）
``` python
!sudo apt install python3-pip
!sudo apt install python3.7-distutils

!python3.7 -m pip install torch==1.0.1
```

* 安裝Python 3.7版本的開發工具
``` python
!sudo apt install python3.7-dev
```


* 進入faster-rcnn.pytorch專案的根目錄，安裝專案的依賴函式庫
``` python
cd('/content/faster-rcnn.pytorch')
!python3.7 -m pip install -r requirements.txt
cd('lib')
!python3.7 setup.py build develop
```


* 下載預訓練模型和資料集，並設定環境以便於後續使用
``` python
cd('..')
!mkdir data
os.chdir('data')
!mkdir pretrained_model
os.chdir('pretrained_model')
# 下載預訓練模型res101
!wget https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth
# 下載預訓練模型vgg16
!wget https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth

os.chdir('../') #返回上一個目錄即data/下
# 下載數據集
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
# 解壓縮
!tar xvf VOCtrainval_06-Nov-2007.tar
!tar xvf VOCtest_06-Nov-2007.tar
!tar xvf VOCdevkit_08-Jun-2007.tar
# 建立連結
!ln -s $VOCdevkit VOCdevkit2007 
```


* 重新命名目錄
``` python
cd('/content/faster-rcnn.pytorch/data')
!rm VOCdevkit2007
!mv VOCdevkit VOCdevkit2007
```

* Git版本控制和編譯
``` python
cd('/content/faster-rcnn.pytorch/data') #成為後續操作的基準目錄
!git clone https://github.com/pdollar/coco.git
cd('coco/PythonAPI') #命令將當前工作目錄coco儲存PythonAPI子目錄
!sed -i 's/python/python3.7/g' Makefile #文件中的所有python字串替換為`python3.7
!make #編譯或目前COCO資料集的Python API，以便在後續的程式碼中使用
```

* 確保環境中安裝了特定版本的資料庫
``` python
!python3.7 -m pip install scipy==1.2.1
!python3.7 -m pip install torch==1.0.1 torchvision==0.2.0
!python3.7 -m pip install pyyaml==5.4.1
```



* 指定使用的參數執行trainval_net.py腳本，進行深度學習模型的
``` python
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
```
