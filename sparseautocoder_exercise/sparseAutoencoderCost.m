function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
fprintf('the size of W1: %d, %d\n', size(W1));
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
fprintf('the size of W2: %d, %d\n', size(W2));
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
fprintf('the size of b1: %d, %d\n', size(b1));
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
fprintf('the size of b2: %d, %d\n', size(b2));

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
m = size(data,2);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%% STEP1 feedforward
X = [ones(1, size(data, 2)); data];
z2 = [b1 W1] * X;
a2 = sigmoid(z2);
z3 = [b2 W2] * [ones(1, size(data, 2)); a2];
a3 = sigmoid(z3);
% fprintf('--------------------------------\n');
% fprintf('the size of a2: %d, %d\n', size(a2));
% fprintf('the size of a3: %d, %d\n', size(a3));

%% sparsity parameters
p = mean(a2, 2);
p
sumSparsity = 0;
for i=1:numel(p)
    sumSparsity = sumSparsity + sparsityParam*log(sparsityParam/p(i)) + (1-sparsityParam)*log((1-sparsityParam)/(1-p(i)));
end

%% backpropagation
delta3 = a3 - data;
delta2 = (W2' * delta3 + beta * (repmat(p, 1, m))) .* (sigmoid(a2) .* (1 - sigmoid(a2)));
% fprintf('--------------------------------\n');
fprintf('the size of delta3: %d, %d\n', size(delta3));
fprintf('the size of delta2: %d, %d\n', size(delta2));
Theta2_grad = delta3 * a2' / m;
Theta1_grad = delta2 * data' / m;
% fprintf('--------------------------------\n');
% fprintf('the size of Theta2_grad: %d, %d\n', size(Theta2_grad));
% fprintf('the size of Theta1_grad: %d, %d\n', size(Theta1_grad));

W1grad = Theta1_grad + lambda * W1;
b1grad = sum(delta2, 2) / m;
W2grad = Theta2_grad + lambda * W2;
b2grad = sum(delta3, 2) / m;

% cost = sum(sum((a3 - data) .^ 2 / 2, 1)) / m;
cost = sum(sum(-data .* log(a3) - (1-data) .* log(1 - a3), 2)) / m;
cost = cost + lambda * (sum(sum(W1 .^ 2,2)) + sum(sum(W2 .^ 2,2))) / (2*m) ...
            + beta * sumSparsity;
fprintf('cost: %d\n', cost);















%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

