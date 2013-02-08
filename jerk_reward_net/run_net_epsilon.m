function [ x, ctrls ] = run_net( Q, W2, W3, seqn, x_last, epsilon  )
%RUN_NET Run one iteration of the network
    % Wi - weights from layer i to i+1
    % seqn - sequence number, i.e. which node in the first layer to
    %        activate
    % x_last - the output of the network from the last iteration


ctrl_weights = Q(seqn,:)';

% choose controller by epsilon greedy
if rand < epsilon
    ctrlr = randsample(size(Q,2),1);
else
    [b,ctrlr] = max(ctrl_weights);
end

% activate this controller
ctrls = zeros(size(Q,2),1);
ctrls(ctrlr,1) = 1;

% use these controllers to turn on alpha motor neurons
alphamns = W2'*ctrls;
x = W3'*[x_last,alphamns']';

end

