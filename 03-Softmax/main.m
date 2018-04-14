clc;
clear;
rng(1);
addpath(genpath('../common'));
check_grad = false;
if check_grad
    X = randn(3,10);
    y = unidrnd(4,1,10);
    w = zeros(3,4);
    ave_error = grad_check(@softmax,w(:),X,y);
    
    fprintf('Testing gradient using forward-differencing...\n');
    order = 1;
    derivativeCheck(@softmax,w(:),order,1,X,y);
    
    fprintf('Testing gradient using central-differencing...\n');
    derivativeCheck(@softmax,w(:),order,2,X,y);
    
    fprintf('Testing gradient using complex-step derivative...\n');
    derivativeCheck(@softmax,w(:),order,3,X,y);
    
    fprintf('\n\n\n');
end

%%
addpath(genpath('../minFunc_2012'));
binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

% Training set info
m=size(train.X,2);
n=size(train.X,1);

% Train softmax classifier using minFunc
options = struct('MaxIter', 200);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
% We only use num_classes-1 columns, since the last column is always assumed 0.
theta = rand(n,num_classes)*0.001;

% Call minFunc with the softmax_regression_vec.m file as objective.
%
% TODO:  Implement batch softmax regression in the softmax_regression_vec.m
% file using a vectorized implementation.
%
tic;
theta(:)=minFunc(@softmax, theta(:), options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);
theta = reshape(theta,n, num_classes);
% theta=[theta, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
[~, I] = max(theta'*train.X,[],1);
accuracy = mean(I == train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
[~, I] = max(theta'*test.X,[],1);
accuracy = mean(I == test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);
%%
function [f,g] = softmax(w,X,y)
[d,n] = size(X);
K = length(unique(y));
Y = zeros(K,n);
I = sub2ind(size(Y), y, 1:n);
Y(I) = 1;
w = reshape(w,d,K);
f = 0;
g = zeros(size(w));

hw = exp(w'*X);
normhw = bsxfun(@rdivide,hw,sum(hw));
f = -sum(sum(Y.*log(normhw)));

g = X*(normhw-Y)';
g = g(:);
end