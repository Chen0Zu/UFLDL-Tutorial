clc;clear;
addpath(genpath('../minFunc_2012'));
addpath('../common');
rng(1);

%%

function f = cnnConvolve(image, W)
% Flip W for use in conv2
W = rot90(W,2);
f = conv2(image,W);
end