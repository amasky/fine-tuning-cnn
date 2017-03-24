# CNN with Chainer  

* Fine-tuning Caffemodel (Python 2)  
[chainer-fine-tuning.ipynb](https://github.com/masaki-y/deep-learning-with-chainer/blob/master/chainer-fine-tuning.ipynb)

* Auto Encoder and Learned Filter Visualization (Python 2 or 3)  
[chainer-auto-encoder.ipynb](https://github.com/masaki-y/deep-learning-with-chainer/blob/master/chainer-auto-encoder.ipynb)

## Examples  

* Prediction of fine-tuned CNN   
![buttercup image in dataset](/examples/finetuning-buttercup.png)
```
# 1| Buttercup    |  90.449%
# 2| Iris         |   7.200%
# 3| Daffodil     |   1.860%
# 4| Sunflower    |   0.279%
# 5| Tigerlily    |   0.156%
```

* Achieved filters by unsupervised learning on AE  
![Visualized Filters](/examples/ae-w-drop-relu-adam-epoch500.png)  

## Dependencies
Python 2 (or 3), [chainer](http://chainer.org/), jupyter, matplotlib, scikit-image, tqdm  

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
