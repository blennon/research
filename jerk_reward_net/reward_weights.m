function [ W ] = reward_weights( W, reward, ctrls_used, learn_rate )
%REWARD_WEIGHTS Summary of this function goes here
%   adjust weights/probabilities for ctrls_used

W = W + learn_rate*reward*ctrls_used;
W(W<0) = 0;
W = W ./ repmat(sum(W,2),1,size(W,2)); % normalize

end

