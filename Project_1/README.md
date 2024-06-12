# README

1. Folder Structure

   Please copy my detectron2 folder and configs folder to original [detectron2](https://github.com/facebookresearch/detectron2).
   Folder structure as the following:

   detectron2 
      |-- configs
      |-- detectron2

2. Execution

   To Execute, please go to configs folder. There are three file: `resnet_train.py`, `vgg_train.py`, `my_train.ipynb`.
   `resnet_train.py` will training with ResNet50 backbone and also doing the evaluation.
   `vgg_train.py` will training with VGG16 backbone and also doing the evaluation.
   If you want to seperate the training and evaluation, please using `my_train.ipynb`.

