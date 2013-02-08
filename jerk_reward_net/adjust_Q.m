function [ Q ] = adjust_Q( Q, reward, ctrls_used, discount, learn_rate, st )
%ADJUST_Q Update the Q function according to SARSA
%   Intended to be used at the end of each episode

% Update Q
r = [zeros(st-1,1); reward];
q = Q(ctrls_used == 1);
q = q + learn_rate*(r+discount*[q(2:end); 0] - q);
Q(ctrls_used == 1) = q;

end

