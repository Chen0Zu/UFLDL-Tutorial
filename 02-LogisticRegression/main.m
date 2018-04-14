clear;clc;
addpath('../common');
rng(1);
%%
plot_sigmoid = false;
plot_sigmoid = true;
if plot_sigmoid
    x = -10:0.1:10;
    plot(x,sigmoid(x));
end

%%
binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);
save('mnist.mat','train','test');
n_train = size(train.X,2);
n_test = size(test.X,2);
d = size(train.X,1);
train.X = [ones(1,n_train);train.X];
test.X = [ones(1,n_test);test.X];

w = zeros(d+1,1);

%%
% check_grad = true;
check_grad = false;
if check_grad

    wTest = w;
    X = train.X(:,1:10);
    y = train.y(1:10);
   
    ave_error = grad_check(@logistic_regression,wTest,X,y);
    fprintf('Testing gradient using forward-differencing...\n');
    order = 1;
    derivativeCheck(@logistic_regression,wTest,order,1,X,y);
    
    fprintf('Testing gradient using central-differencing...\n');
    derivativeCheck(@logistic_regression,wTest,order,2,X,y);
    
    fprintf('Testing gradient using complex-step derivative...\n');
    derivativeCheck(@logistic_regression,wTest,order,3,X,y);
    
    fprintf('\n\n\n');
end

options = struct('MaxIter', 200);
[w, ~, ~, output] = minFunc(@logistic_regression, w, options, train.X, train.y);
plot(output.trace.fval);

%%
y_train = sigmoid(w'*train.X);
idx = y_train > 0.5;
y_train(idx) = 1;
y_train(~idx) = 0;
acc_train = mean(y_train == train.y);
y_test = sigmoid(w'*test.X);
idx = y_test > 0.5;
y_test(idx) = 1;
y_test(~idx) = 0;
acc_test = mean(y_test == test.y);

fprintf('Training accuracy %f\n', acc_train);
fprintf('Testing accuracy %f\n', acc_test);

function h = sigmoid(z)
h = 1./(1+exp(-z));
end

function [f,g] = logistic_regression(w,X,y)

f = -sum(y.*log(sigmoid(w'*X)) + (1-y).*log(1-sigmoid(w'*X)));
g = X*(sigmoid(w'*X)-y)';

end