clear; clc;
% 加载模型
net = importNetworkFromPyTorch("cnnnet.pt");

% 加上一个输入层
InputSize = [224 224 3];
inputLayer = imageInputLayer(InputSize,Normalization="none");
net = addInputLayer(net,inputLayer,Initialize=true);
analyzeNetwork(net)

% 测试图像
Im = imread("CatDogs\test\cat\cat.0.jpg");
Im = imresize(Im,[224,224]);
imshow(Im)
Im = rescale(Im,0,1);
meanIm = [0.5 0.5 0.5];
stdIm = [0.5 0.5 0.5];
Im = (Im - reshape(meanIm,[1 1 3]))./reshape(stdIm,[1 1 3]);
Im_dlarray = dlarray(single(Im),"SSCB");
% squeezeNet = squeezenet;
% ClassNames = squeezeNet.Layers(end).Classes;
ClassNames = categorical(["cat","dog"]);
prob = predict(net,Im_dlarray);
[~,label_ind] = max(prob);
ClassNames(label_ind)

