# Face-Recognition-Flask-GUI
This repo is a face recognition demonstration system, which is a part of my capstone project at Hong Kong Polytechnic University. This system is developed by Flask, Pytorch, HTML, JavaScript, and CSS. Available methods include EigenFace, LBP, and ResNet-based deep face recognition.

For deep face recognition, the supported loss functions include:

1. loss=softmax: the standard Softmax loss 
2. loss=asoftmax: proposed in SphereFace [1]
3. loss='amsoftmax': proposed in AMSoftmax [2] and CosineFace [3]

## User Interface

<p align="center">
  <img src="https://github.com/aaronzguan/Face-Recognition-Flask-GUI/blob/master/UI.png" height="450">
</p>

To understand how it works, please watch the [Demonstration Video](https://www.youtube.com/watch?v=DF9S3HiIlSo)

## Reference
1. W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song, "SphereFace: Deep Hypersphere Embedding for Face Recognition," CVPR, 2017.
2. F. Wang, J. Cheng, W. Liu, and H. Liu, "Additive Margin Softmax for Face Verification," IEEE Signal Processing Letters, vol. 25, no. 7, pp. 926-930, 2018.
3. H. Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition," CVPR, 2018.
