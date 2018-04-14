function ave_error = grad_check(fun,x,varargin)
epsilon = 1e-4;
d = length(x);
[f,g] =fun(x,varargin{:});
g_est = zeros(size(g));
for i = 1:d
    T = x;
    T0 = T; T0(i) = T(i) - epsilon;
    T1 = T; T1(i) = T(i) + epsilon;
    f0 = fun(T0, varargin{:});
    f1 = fun(T1, varargin{:});
    g_est(i) = (f1-f0)/(2*epsilon);
end
ave_error = mean(abs(g - g_est));
fprintf("The ave_error is %e\n", ave_error);
disp([g,g_est]);
end