clear all

%% Parameters
% model
patchSize = 40;
depth = 17;
channels = 1;

% dataset
trainDataPath = 'E:\Code\MATLAB\DnCNN\testsets\BSD68';
validateImage = 'E:\Code\MATLAB\DnCNN\testsets\Set12\08.png';
noiseStd = 15/255; % sigma
patchesPerImage = 512;

% trainning
initialLearnRate = 0.001;
learnRateSchedule = 'piecewise';
learnRateDropFactor = 0.5;
learnRateDropPeriod = 3;
momentum = 0.9;
l2Regularization = 0.0001; % Weight decay
batchSize = 64;
epochs = 30;
shuffle = 'every-epoch';
checkpointPath = '.\CheckPoints';

% output
outputPath = ['dncnn_sigma-' num2str(floor(noiseStd*255)) '_' datestr(datetime('now'), 'mm_dd__HH_MM') '.mat'];

rng(2017) % For reproducibility

%% Load validate dataset
data_src = imread(validateImage);
data_src = im2single(data_src);
%data_noise = imnoise(validate_src,'gaussian', 0, noiseStd.^2);
data_noise = noiseStd * randn(size(data_src), 'single');
data_noisy = data_noise + data_src;

data_height = size(data_src, 1);
data_width = size(data_src, 2);
patch_num_v = floor(data_height / patchSize);
patch_num_h = floor(data_width / patchSize);
validate_noisy = zeros(patchSize, patchSize, channels, patch_num_v*patch_num_h, 'single');
validate_noise = zeros(patchSize, patchSize, channels, patch_num_v*patch_num_h, 'single');
for i = 1:patch_num_h
    for j = 1:patch_num_v
        validate_noisy(:, :, :, j+(i-1)*patch_num_v) = data_noisy(1+(j-1)*patchSize:j*patchSize, 1+(i-1)*patchSize:i*patchSize);
        validate_noise(:, :, :, j+(i-1)*patch_num_v) = data_noise(1+(j-1)*patchSize:j*patchSize, 1+(i-1)*patchSize:i*patchSize);
    end
end

%% Define network
net = model(patchSize, channels, depth);
%net = dnCNNLayers('NetworkDepth', depth);

%% Prepare dataset for trainning
imds = imageDatastore(trainDataPath);
fileNums = numel(imds.Files);
% "GaussianNoiseLevel" specifies the standard derivation rather than variance of AGWN in function denoisingImageSource
source = denoisingImageSource(imds, ...
    'patchSize', patchSize, ...
    'ChannelFormat', 'GrayScale', ...
    'GaussianNoiseLevel', noiseStd, ...
    'PatchesPerImage', patchesPerImage);

%% Train
% Trainning options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearnRate, ...
    'Momentum', momentum, ...
    'LearnRateSchedule', learnRateSchedule, ...
    'LearnRateDropFactor', learnRateDropFactor, ...
    'LearnRateDropPeriod', learnRateDropPeriod, ...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs', epochs, ...
    'L2Regularization', l2Regularization, ...
    'Shuffle', shuffle, ...
    'CheckpointPath', checkpointPath, ...
    'Plots', 'training-progress', ...
    'Verbose', 1, ...
    'VerboseFrequency', floor(fileNums*patchesPerImage/batchSize), ...
    'ValidationData', {validate_noisy, validate_noise}, ...
    'ValidationFrequency', floor(fileNums*patchesPerImage/batchSize/4), ...
    'ValidationPatience', Inf, ...
    'ExecutionEnvironment', 'gpu');

trainedNet = trainNetwork(source, net, options);

%% Save Trained Network
save(outputPath, 'trainedNet')