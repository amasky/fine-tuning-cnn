## Deep Learning with Chainer  

* Deep Convolutional Neural Networks with Chainer  
https://github.com/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-CNN-CIFAR10.ipynb

* Fine-tuning (Transfer Learning) Caffemodel with Chainer  
https://github.com/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-Fine-Tuning.ipynb

## Dependencies
Python 2.7 (3.5: preparing), cython, Chainer (v1.5), Jupyter, matplotlib, progressbar2  
If you don't have these python packages, I recommend you install them by "pip install".

## Usage
Open a Jupyter's session in your browser.  
Then select the `Run All` from the `cell` in the top menu.  
```sh
$ jupyter notebook xxx.ipynb
```

If you use a GPU, set the GPU ID in `gpu`.
Chainer will switch automatically to GPU mode.
```py
gpu = -1 # gpu device ID (cpu if this negative)
xp = cuda.cupy if gpu >= 0 else np  
```
