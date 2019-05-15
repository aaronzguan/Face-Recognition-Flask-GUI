# Face-Recognition-Flask-GUI
This repo is a face recognition demonstration system, which is a part of my capstone project at Hong Kong Polytechnic University. This system is developed by Flask, Pytorch, HTML, JavaScript, and CSS. Available methods include EigenFace, LBP, and ResNet-based deep face recognition.

## Features

This system can support real-time face capture, alignment, and detection from a webcam. MTCNN [1] in PyTorch is used for facial landmarks detection. User is able to register his/her face into the database in case that the database does not have his/her face. The 3 most similar faces in the database will show up determined by the largest cosine similarity.

For Eigenface, the 100 most significant eigenvectors are used for face recognition. It can show the average face in the database and also the reconstructed uploaded face by using the 100 eigenvectors. For LBP, the 3x3 window is considered for extracting non-uniform LBP features. For deep face recognition, this system support resnet-10, resnet-20, and resnet-64 network models. 

The supported loss functions include:

1. loss=softmax: the standard Softmax loss 
2. loss=asoftmax: proposed in SphereFace [2]
3. loss='amsoftmax': proposed in AMSoftmax [3] and CosineFace [4]

## User Interface

<p align="center">
  <img src="https://github.com/aaronzguan/Face-Recognition-Flask-GUI/blob/master/UI.png" height="450">
</p>

To understand how it works, please watch the [Demonstration Video](https://www.youtube.com/watch?v=DF9S3HiIlSo).

## Reference
1. K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multi-task cascaded convolutional networks,” CoRR, vol. abs/1604.02878, 2016.
2. W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song, "SphereFace: Deep Hypersphere Embedding for Face Recognition," CVPR, 2017.
3. F. Wang, J. Cheng, W. Liu, and H. Liu, "Additive Margin Softmax for Face Verification," IEEE Signal Processing Letters, vol. 25, no. 7, pp. 926-930, 2018.
4. H. Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition," CVPR, 2018.
