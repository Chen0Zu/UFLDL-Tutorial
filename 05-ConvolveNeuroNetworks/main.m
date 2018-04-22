clc;clear;
addpath(genpath('../minFunc_2012'));
addpath('../common');
rng(1);

test = [1 1 1 0 0; 0 1 1 1 0; 0 0 1 1 1; 0 0 1 1 0; 0 1 1 0 0];
W = [1 0 1; 0 1 0; 1 0 1];
f = cnnConvolve(test,W);

%%

function f = cnnConvolve(image, W)
% Flip W for use in conv2
W = rot90(W,2);
f = conv2(image,W,'valid');
end