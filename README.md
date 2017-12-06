# DnCNN
Implementation of DnCNN in MATLAB using Neural Network Toolbox™

Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing.
https://github.com/cszn/DnCNN


TODO:
1. Data augmentation in function denoisingImageSource (flip, rotate)
2. Other optimizers (Adam, etc. This may be done by modifying ```toolbox\nnet\cnn\+nnet\+internal\+cnn\Trainer.m```. I will make a try when I have time.)