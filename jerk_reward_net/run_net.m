function [ x, ctrls ] = run_net( W1, W2, W3, seqn, n, x_last  )
%RUN_NET Run one iteration of the network
    % Wi - weights from layer i to i+1
    % seqn - sequence number, i.e. which node in the first layer to
    %        activate
    % n - number of controllers to randomly activate
    % x_last - the output of the network from the last iteration


% choose a random subset of n controllers based on the learned 'weights'
% from the sequence node (first) layer to the controllers layer
ctrl_weights = W1(seqn,:)';

rnd_ctrls = randsample(size(W1,2),n,true,ctrl_weights);
ctrls = zeros(size(W1,2),1);

% activate these controllers fully
ctrls(rnd_ctrls,1) = 1;

% use these controllers to turn on alpha motor neurons
alphamns = W2'*ctrls;
x = W3'*[x_last,alphamns']';

end

