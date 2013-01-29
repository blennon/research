function [ W ] = bias_weights( W, bias, half_span )
%BIAS_WEIGHTS Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(W);
[C,I] = max(W,[],2);
for i=1:m
    if I(i)+half_span > n
        indcs = I(i)-half_span:n;
    elseif I(i) - half_span < 1
        indcs = 1:I(i)+half_span;
    else
        indcs = I(i)-half_span:I(i)+half_span;
    end
    W(i,indcs) = W(i,indcs) + bias;
end
W = W ./ repmat(sum(W,2),1,size(W,2)); % normalize
end

