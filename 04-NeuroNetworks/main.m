clc;
clear;
%%
n = 100; d = 28;
X = rand(n,d);
n_input = d; n_hidden = 16; n_output = 3;
W1 = zeros(n_hidden,n_input);
b1 = zeros(n_hidden,1);
W2 = zeros(n_output,n_hidden);
b2 = zeros(n_output,1);

%%

function [f,g] = loss()
end

function f = forward()
end

function [f,g] = MLP(theta, X, y)

end
