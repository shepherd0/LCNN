Local Contrastive Neural Network

Tensorflow implementation of Local Contrastive Neural Network  by Chuan Xu et al.

Prerequisites

Python 3.0+
NumPy
SciPy
tqdm
tenorpack 0.8.0
Tensorflow r1.4+
Pillow
matplotlib
scikit-image
h5py
 
Data

Omniglot

Preparation

python  LCL_Omniglot.py  --task=prepare_data


Omniglot is downloaded from https://github.com/brendenlake/omniglot.git, and a few pickle files are created,for example tiny228_omni.pkl. The images are resized to shape (28, 28). 90-degree-transforms (0, 90, 180, 270) are used for creating novel sample classes.



 
Train

python  LCL_Omniglot.py --task=train --train_dataset=omniglot_tiny2 --test_dataset=omniglot_oneshot -n=20 --gpu=0,1,2,3 --batch_size=40 --drop_1=700 --drop_2=800 --drop_3=900 --max_epoch=900   

if training in on CPU, please remove parameter  --gpu=0,1,2,3, here 0,1,2,3 are numbers of GPU on case of multiple GPUs. if on 2 Nvidia K80 GPUs, the training with 900 epochs take about 30 hours.
 
Test

python  LCL_Omniglot.py  --task=test  --test_dataset=omniglot_oneshot -n=20  --gpu=0  --batch_size=40  --test_times=100       

The model was trained on 60 characters and 20 samples each character, which can directly output one-shot classification accuracy 97.99¡À0.05%. Please download model from https://pan.baidu.com/s/1smCndbJ to code directory and unzip it.
 
python  LCL_Omniglot.py  --task=test  --test_dataset=omniglot_oneshot -n=20  --gpu=0  --batch_size=40  --test_times=100   --load=model-57216

Notes

 
Resources

This code has only been tested on Ubuntu 16.4 and Window 2007.