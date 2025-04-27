clear; clc;
% 下载模型，如果下载失败试着连一下外网
modelfile = matlab.internal.examples.downloadSupportFile("nnet", "data/PyTorchModels/mnasnet1_0.pt");

% 加载模型
net = importNetworkFromPyTorch("modelfile");
% net = importNetworkFromPyTorch("mnasnet1_0.pt");

% 可视化
analyzeNetwork(net) 

% 加上一个输入层
InputSize = [224 224 3];
inputLayer = imageInputLayer(InputSize,Normalization="none");
net = addInputLayer(net,inputLayer,Initialize=true);
analyzeNetwork(net)

% 识别单张图片
Im = imread("peppers.png");
Im = imresize(Im,[224,224]);
imshow(Im)
Im = rescale(Im,0,1);
meanIm = [0.485 0.456 0.406];
stdIm = [0.229 0.224 0.225];
Im = (Im - reshape(meanIm,[1 1 3]))./reshape(stdIm,[1 1 3]);
Im_dlarray = dlarray(single(Im),"SSCB");
squeezeNet = squeezenet;
ClassNames = squeezeNet.Layers(end).Classes;
prob = predict(net,Im_dlarray);
[~,label_ind] = max(prob);
ClassNames(label_ind)





