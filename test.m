%% Parameters
imagePath = 'E:\Code\MATLAB\DnCNN\testsets\Set12\09.png';
noiseStd = 15/255; % sigma
netPath = '.\Records\1\dncnn_sigma-15_11_29__14_51.mat';

%% Read test image and add AGWN
I = imread(imagePath);
noisyI = imnoise(I,'gaussian', 0, noiseStd.^2);

%% Read net
load(netPath)

%% Denoise
denoisedI = denoiseImage(noisyI, trainedNet);

%% Show denoised Result
figure, imshow([I, noisyI, denoisedI])
%figure, imshowpair(noisyI, denoisedI, 'diff'), title('Method noise')
%figure, imshowpair(I, denoisedI, 'diff'), title('Degradation between source and denoised')

%% Calculate PSNR
fprintf('PSNR between source and noisy input:   %f\n', psnr(I, noisyI));
fprintf('PSNR between source and denoised:      %f\n', psnr(I, denoisedI));
%fprintf('PSNR between noisy input and denoised: %f\n', psnr(noisyI, denoisedI));