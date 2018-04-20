clc;
clear;
dbstop if error
addpath('../common');
addpath(genpath('../minFunc_2012'));
rng(2);
%%
n = 100; d = 4;
X = rand(d, n);
n_input = d; n_hidden = 3; n_output = 4;
W1 = randn(n_hidden,n_input);
b1 = randn(n_hidden,1);
W2 = randn(n_output,n_hidden);
b2 = randn(n_output,1);

theta = [W1(:);b1;W2(:);b2];

check_grad = true;
if check_grad
    n = 10; d = 4;
    X = rand(d, n);
    y = unidrnd(4,1,n);
    n_input = d; n_hidden = 3; n_output = 4;
    W1 = randn(n_hidden,n_input);
    b1 = randn(n_hidden,1);
    W2 = randn(n_output,n_hidden);
    b2 = randn(n_output,1);
    
    theta = [W1(:);b1;W2(:);b2];
    ave_error = grad_check(@MLP,theta,X,y,n_input, n_hidden, n_output);
    
    fprintf('Testing gradient using forward-differencing...\n');
    order = 1;
    derivativeCheck(@MLP,theta,order,1,X,y,n_input, n_hidden, n_output);
    
    fprintf('Testing gradient using central-differencing...\n');
    derivativeCheck(@MLP,theta,order,2,X,y,n_input, n_hidden, n_output);
    
    fprintf('Testing gradient using complex-step derivative...\n');
    derivativeCheck(@MLP,theta,order,3,X,y,n_input, n_hidden, n_output);
    
    fprintf('\n\n\n');
end

%%

%%
function [W1,b1,W2,b2] = unfold(theta, lambda, n_input, n_hidden, n_output)
W1 = reshape(theta(1:n_hidden*n_input), n_hidden, n_input);
b1 = theta(n_hidden*n_input+1:n_hidden*n_input+n_hidden);
W2 = reshape(theta(n_hidden*n_input+n_hidden+1:...
    n_hidden*n_input+n_hidden+n_output*n_hidden), n_output, n_hidden);
b2 = theta(n_hidden*n_input+n_hidden+n_output*n_hidden+1:...
    n_hidden*n_input+n_hidden+n_output*n_hidden+n_output);
end

function f = sigmoid(x)
f = 1./(1+exp(-x));
end

function [f,g] = MLP(theta, X, y, n_input, n_hidden, n_output)
[W1,b1,W2,b2] = unfold(theta, n_input, n_hidden, n_output);
n = size(X,2);
K = length(unique(y));
Y = zeros(K,n);
I = sub2ind(size(Y), y, 1:n);
Y(I) = 1;

%% forward
a1 = X;
z2 = bsxfun(@plus, W1*a1, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);
% loss
expz3 = exp(z3);
p = bsxfun(@rdivide, expz3, sum(expz3));
f = -sum(sum(Y.*log(p)));

%% backward

delta3 = -(Y - p);
W2_grad = delta3*a2';
b2_grad = sum(delta3,2);
delta2 = W2'*delta3.*(a2.*(1-a2));
W1_grad= delta2*a1';
b1_grad = sum(delta2,2);

%%
g = [W1_grad(:);b1_grad;W2_grad(:);b2_grad];
end
