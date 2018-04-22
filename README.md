# tf-dagmm
A tensorflow Implementation of [dagmm](https://openreview.net/pdf?id=BJJLHbb0-): Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection, Zong et al, 2018. Furthermore, I built the convolutional autoencoder to analyze the data of images instead of fully-connected autoencoder.

## Requirement
I've trained the model successfully on the below packages:
* Python 3.6.4
* Tensorflow 1.5.0(GPU version)

## Encoded to a vector of higher dimension
The auther encode the input to a vector of a very low dimension(only 1 or 2). This is too small since I am doing
the anomaly detection for images with the size `128 x 128`. I use a meta dense layer to map the encoded vector into
the into the low dimension space. Therefore, I could have a better reconstruction error and a trainable GMM as well.  

## Note
* This tf model is used to perform a real case of anomaly detection for my job. 
I cannot provide the dataset I used due to the commercial security.
* I've done some experiments on using several autoencoders(compressions namely in the paper) to analyze the important regions as experts suggested in each image.
The results didn't perform well however.
* In my case the required training time is short. The results seem good in less than 2000 epochs

## Train
Please take a look at `train.py`, example script:
```
$ python ./train.py --encoded_dims 3 --mixtures 7

```
## Issue
I cannot avoid the singularity issue of gmm, even if the penality term mention in the paper is added. 
Therefore, I added some jitter to the diagonal of the covariance matrix. Not sure it's a good solution or not.  


## Result
some pictures of results  
![Alt text](pics/hist.png?raw=true "histogram")
![Alt text](pics/test_scatter.png?raw=true "scatter plot")
![Alt text](pics/tsne.png?raw=true "tsne")
