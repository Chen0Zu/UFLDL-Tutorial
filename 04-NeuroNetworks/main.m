clc;
clear;
dbstop if error
%%
n = 100; d = 4;
X = rand(n,d);
n_input = d; n_hidden = 3; n_output = 2;
W1 = randn(n_hidden,n_input);
b1 = randn(n_hidden,1);
W2 = randn(n_output,n_hidden);
b2 = randn(n_output,1);

theta = [W1(:);b1;W2(:);b2];

[Z1,z1,Z2,z2] = unfold(theta, n_input, n_hidden, n_output);

%%

%%
function [W1,b1,W2,b2] = unfold(theta, n_input, n_hidden, n_output)
W1 = reshape(theta(1:n_hidden*n_input), n_hidden, n_input);
b1 = theta(n_hidden*n_input+1:n_hidden*n_input+n_hidden);
W2 = reshape(theta(n_hidden*n_input+n_hidden+1:...
    n_hidden*n_input+n_hidden+n_output*n_hidden), n_output, n_hidden);
b2 = theta(n_hidden*n_input+n_hidden+n_output*n_hidden+1:...
    n_hidden*n_input+n_hidden+n_output*n_hidden+n_output);
end

function [f,g] = loss()
end

function f = forward()
end

function [f,g] = MLP(theta, X, y)

end
