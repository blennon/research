function [ Q ] = adjust_Q_QL( Q, reward, ctrls_used, discount, learn_rate, st )
%ADJUST_Q Update the Q function according to SARSA
%   Intended to be used at the end of each episode

% Update Q
r = [zeros(st-1,1); reward];
qmax = max(Q,[],2);
q = Q(ctrls_used == 1);
q = q + learn_rate*(r+discount*[qmax(2:st); 0] - q);
Q(ctrls_used == 1) = q;

end

