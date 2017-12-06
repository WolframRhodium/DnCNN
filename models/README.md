Internal 'denoiseImage.m' should be modified to support DAGNetworks like ResNet:

Replace L73:
```MATLAB
res = activations(net,inputImage,numLayers-1,'OutputAs','channels');
```
with
```MATLAB
if isa(net,'DAGNetwork')
    res = activations(net,inputImage,'Output');
else
    res = activations(net,inputImage,numLayers-1,'OutputAs','channels');
end
```

In order to use custom layers, L58:
```MATLAB
validateInputNetwork(net);
```
have to be removed.
