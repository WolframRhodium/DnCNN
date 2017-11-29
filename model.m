function layers = model(patchSize, channels, networkDepth)
    layers = [];
    b_min = 0.025;

    % Input
    input = imageInputLayer([patchSize patchSize channels], ...
        'Normalization', 'none', ...
        'Name', 'Input');

    layers = [layers, input];
    
    % Conv+ReLU
    conv = convolution2dLayer(3, 64, ...
        'Stride', 1, ...
        'Padding', 'same', ...
        'WeightLearnRateFactor', 1, ...
        'WeightL2Factor', 1, ...
        'BiasLearnRateFactor', 1, ...
        'BiasL2Factor', 0, ...
        'Name', 'Conv1');
    conv.Weights = sqrt(2/(9*64))*randn(3,3,channels,64,'single');
    conv.Bias = zeros(1,1,64,'single');

    relu = reluLayer(...
        'Name', 'ReLU1');

    layers = [layers, conv, relu];
    
    % Conv+BN+ReLU
    for i = 2:networkDepth-1
        conv = convolution2dLayer(3, 64, ...
            'Stride', 1, ...
            'Padding', 'same', ...
            'WeightLearnRateFactor', 1, ...
            'WeightL2Factor', 1, ...
            'BiasLearnRateFactor', 0, ...
            'BiasL2Factor', 0, ...
            'Name', ['Conv', num2str(i)]);
        conv.Weights = sqrt(2/(9*64))*randn(3,3,64,64,'single');
        conv.Bias = zeros(1,1,64,'single');

        bnorm = batchNormalizationLayer(...
            'Scale', clipping(sqrt(2/(9*64))*randn(1,1,64,'single'), b_min), ...
            'ScaleLearnRateFactor', 1, ...
            'ScaleL2Factor', 0, ...
            'Offset', zeros(1,1,64,'single'), ...
            'OffsetLearnRateFactor', 1, ...
            'OffsetL2Factor', 0, ...
            'Name', ['BNorm', num2str(i)]);

        relu = reluLayer(...
            'Name', ['ReLU', num2str(i)]);

        layers = [layers, conv, bnorm, relu]; %#ok<AGROW>
    end
    
    % Conv
    conv = convolution2dLayer(3, channels, ...
        'Stride', 1, ...
        'Padding', 'same', ...
        'WeightLearnRateFactor', 1, ...
        'WeightL2Factor', 1, ...
        'BiasLearnRateFactor', 1, ...
        'BiasL2Factor', 0, ...
        'Name', ['Conv', num2str(networkDepth)]);
    conv.Weights = sqrt(2/(9*64))*randn(3,3,64,channels,'single');
    conv.Bias = zeros(1,1,channels,'single');

    layers = [layers, conv];
    
    % Regression
    regression = regressionLayer(...
        'Name', 'Output');

    layers = [layers, regression];
end

function A = clipping(A, b)
    A(A>=0 & A<b) = b;
    A(A<0 & A>-b) = -b;
end