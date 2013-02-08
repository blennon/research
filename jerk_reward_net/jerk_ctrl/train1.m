% This neural network learns a correct sequence of controllers which
% directly control the systems jerk to move a point mass from x0 to xf in
% sf time steps, ending with zero velocity.

clear all;
close all;
clc;

% PARAMETERS
sn = 10;            % max number of sequence points
ctrls = -.2:.1:.2;    % controller values
iters = 10000;      % Number of training iterations
state0 = [0 0 0 0]; % starting state of network (p,v,a,j)
xf = 5.4;           % final location of point mass
sf = 10;            % desired final sequence point
vf = 0;             % desired final velocity
vmax = 1;           % maximum absolute velocity
amax = .5;          % maximum absolute acceleration
statef = [xf vf 0 0];
discount = .95;     % amount to discount the rewards
epsilon = .01;      % epsilon greedy parameter
learn_rate = .1;    % learning rate
beta = 1;           % explore/exploit parameter

% REWARD COEFFICIENTS
cx = log(.1)/(.9*xf); % coeff for .1 reward when .9*xf from xf
cv = log(.1)/.9;      % coef for .1 reward when final velocity is .9 
cj = log(.1)/.9;      % jerk error coefficients
rc = [cx,cx,0,0];

net = Net(state0, ctrls, vmax, amax, .2);
actor = Actor(size(ctrls,2),sn);

for t=1:iters
    net.state = state0;
    state_monitor = StateMonitor(state0,sn);
    beta = beta*1.0005;
    for st=1:sn
        
        c = actor.choose_controller(st, beta);
        state = net.run(c);
        state_monitor.record(state, st);
        
        if st == sn
            if state(1) < 0
                reward = -.1;
            else
                reward = exp(rc*abs(statef-state)');
            end
            actor.update_Q_end(reward, learn_rate);
            break
        end
        
    end
    if mod(t,500) == 0
        state_monitor.plot(1)
        actor.plot_Q(2)
        actor.plot_policy(beta,3)
    end
end
actor.plot_Q(2)
