# Deep Learning with Chainer  

* Deep Convolutional Neural Networks (Python 2 or 3)  
[nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-CNN-CIFAR10.ipynb](http://nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-CNN-CIFAR10.ipynb)

* Fine-tuning Caffemodel (Python 2)  
[nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-Fine-Tuning.ipynb](http://nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-Fine-Tuning.ipynb)

* Auto Encoder and Learned Filter Visualization (Python 2 or 3)  
[nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-Auto-Encoder.ipynb](http://nbviewer.jupyter.org/github/masaki-y/Deep-Learning-with-Chainer/blob/master/Chainer-Auto-Encoder.ipynb)

## Examples  
* Prediction of CNN  
![ship image in CIFAR-10](/examples/cifar10-ship.png)
```
# 1| ship         |  89.769%
# 2| automobile   |  10.231%
# 3| airplane     |   0.000%
# 4| truck        |   0.000%
# 5| cat          |   0.000%
```
* Feature maps of the 1st layer  
![feature maps](/examples/cifar10-fmap.png)

* Training CNN  
![training loss](/examples/cifar10-loss.png)  

* Prediction of fine-tuned CNN   
![buttercup image in dataset](/examples/finetuning-buttercup.png)
```
# 1| Buttercup    | 94.995%
# 2| Iris         |  3.462%
# 3| Daffodil     |  1.258%
# 4| Tigerlily    |  0.130%
# 5| Cowslip      |  0.120%
```

* Achieved filters by unsupervised learning on AE  
![Visualized Filters](/examples/ae-w-drop-relu-adam-epoch500.png)  

## Dependencies
Python 2 or 3, [chainer](http://chainer.org/) (v1.5 or later), jupyter, matplotlib, scikit-image, tqdm  

## Usage
Open a Jupyter's session in your browser.  
```shellsession
➜  jupyter notebook Chainer-CNN-CIFAR10.ipynb
```
Then select the `Run All` from the `cell` in the top menu.  

If you use a GPU, set the GPU ID to `gpuid` in the jupyter notebook.
Chainer will switch automatically to GPU mode.
```py
gpuid = -1 # gpu device ID (cpu if this negative)
xp = cuda.cupy if gpuid >= 0 else np  
```
