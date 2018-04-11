function main
clc;clear;
addpath(genpath('../minFunc_2012'));
rng(1);

check_grad = true;
check_grad = false;

data = load('housing.data');
n = length(data);
data = [ones(n,1),data];
data = data(randperm(n),:);

trainX = data(1:400,1:end-1);
trainY = data(1:400,end);
testX = data(401:end,1:end-1);
testY = data(401:end,end);

[m,d] = size(trainX);
w = rand(d,1);

if check_grad

    wTest = w;
    X = trainX;
    y = trainY;
    
    fprintf('Testing gradient using forward-differencing...\n');
    order = 1;
    derivativeCheck(@linear_regression,wTest,order,1,X,y);
    
    fprintf('Testing gradient using central-differencing...\n');
    derivativeCheck(@linear_regression,wTest,order,2,X,y);
    
    fprintf('Testing gradient using complex-step derivative...\n');
    derivativeCheck(@linear_regression,wTest,order,3,X,y);
    
    fprintf('\n\n\n');
end

options = struct('MaxIter', 200);
[w, ~, ~, output] = minFunc(@linear_regression, w, [], trainX, trainY);
plot(output.trace.fval);

train_rms = sqrt(mean((trainX*w-trainY).^2));
test_rms = sqrt(mean((testX*w-testY).^2));
fprintf('RMS training error: %f\n', train_rms);
fprintf('RMS testing error: %f\n', test_rms);

% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(testY);
  predicted_prices = testX*w;
  predicted_prices=predicted_prices(I);
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end
end

function [fvalue, grad] = linear_regression(w, X, y)

fvalue = 0.5*sum((X*w-y).^2);
grad = X'*(X*w-y);
end



