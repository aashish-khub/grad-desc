function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

% ====================== YOUR CODE HERE ======================
% Instructions: Perform a single gradient step on the parameter vector
%               theta.
%
% Hint: While debugging, it can be useful to print out the values
%       of the cost function (computeCost) and gradient here.
%

%delta_theta = zeros(size(theta));
hX = X*theta;
errorThisIter = hX - y;
for j = 1:(length(theta)),
s = 0;
thisJx = X(:,j);
for i = 1:m,
s += (hX(i) - y(i))*thisJx(i);
end;
theta(j) -= (alpha/m) * s;

%delta_theta(j) += errorThisIter' * thisJx;  %'I have no idea why this failed to work but whatever

%theta -= (alpha/m) * delta_theta;

% ============================================================

% Save the cost J in every iteration
J_history(iter) = computeCost(X, y, theta);

end

end
