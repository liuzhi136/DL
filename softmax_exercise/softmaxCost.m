function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
[feats, m] = size(data);
temp = zeros(m, numClasses);
for i=1:numClasses
    temp(:, i) = (labels == i);
end
y = temp;
fprintf('the size of y: %d, %d\n', size(y));
theta = reshape(theta, numClasses, inputSize);
fprintf('the size of theta: %d, %d\n', size(theta));

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
% X = [ones(1, m) data];
X = data;
% theta = [rand(numClasses, 1), theta];
fprintf('the size of X: %d, %d\n', size(X));
% fprintf('the size of theta: %d, %d\n', size(theta));
tmp = theta * X;
tmp = bsxfun(@minus, tmp, max(tmp, [], 1));
h = exp(tmp);
fprintf('the size of h: %d, %d\n', size(h));
sumH = sum(h, 1);
h = bsxfun(@rdivide, h, sumH);

cost = -sum(sum(y' .* log(h), 1)) / m + lambda * sum(sum(theta .^ 2, 2)) / 2;
fprintf('the size of cost: %d\n', cost);
thetagrad = -(y' - h) * X' / m + lambda * theta;










% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
