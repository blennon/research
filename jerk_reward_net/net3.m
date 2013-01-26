% This is a simple neural network where the correct sequence of controllers
% is to be learned in order to move a point mass from x0 to xf

% RESULTS: This network learns to move the mass from x0 to xf in a straight
% line in desired number of steps.  The reward is based on a moving average
% of past performance and thus serves as a dynamic feedback.

clear all;
close all;
clc;

%% Instantiate net

% PARAMETERS
sn = 20; % max number of sequence points
g = .1; % granularity of spacing between W2 weights (i.e. granularity of control of alpha motor neurons)
alpha_max = 1; % maximum output value of alpha motor neurons
w2_max = 1; % maximum weight value impinging on the alpha MNs
nctrlrs = size(0:g:w2_max,2); % number of controllers affecting the alpha MNs
                               % this represents only values in {0,.1,...,1.0}
                               % for controlling the value of alpha MNs


% WEIGHTS -- (source x target)
W1 = 1/nctrlrs * ones(sn,nctrlrs); % probability distbns of choosing ctrlr i at sequence point n.
W2 = [transpose(0:g:w2_max),zeros(size(0:g:w2_max,2),1)];
W3 = [1,1,-1]';


%% Train the network

iters = 3000;   % Number of training iterations
tm = 100;       % Plot monitor every tm iterations
x0 = 0.0;       % starting location of point mass
xf = 3.0;       % final location of point mass
sf = 10;        % desired final sequence point

% LEARNING RATES
learn_rate = .1;
sig = @(x,ymax,ymin,xmin,xmax,beta) (ymax-ymin)./(1+exp(-(x - (xmax-xmin)/2)/beta)) + ymin;
% distance error coefficients
c1_start = log(.1)/(.9*xf); % coeff for .1 reward when .9*xf from xf
c1_end = log(.1)/(.1*xf);   % coeff for .1 reward when .1*xf from xf
c1 = @(xdavg) sig(xdavg,c1_start,c1_end,0,xf,xf/8);
% sequence length error coefficients
c2_start = log(.1)/(.9*sf); % coeff for .1 reward when .9*sf from sf
c2_end = log(.1)/(.1*sf);   % coeff for .1 reward when .1*sf from sf
c2 = @(sdavg) sig(sdavg,c2_start,c2_end,0,sf,sf/8);

% moving average parameters
sma_coef = .95;     % s_dist moving average coefficient
xma_coef = .95;     % x_dist moving average coefficient
s_dist_ma = sf;     % s_dist MA starting value
x_dist_ma = xf;     % x_dist MA starting value
target_reward_ma = 0;
trma_coef = .9;
jerk_ma = 0;
jma_coef = .6;

% MONITORING
jerk_history = [];
s_dist_history = [];
x_dist_history = [];
reward_history = zeros(5,iters);

% IMPLICIT VARIABLES LEGEND
% st - current sequence step number
% xt - current value of x for step i
% t - training iteration

for t=1:iters
    
    xt = 0;
    new_xt = 0;
    ctrls_used = zeros(sn,nctrlrs);
    traj = [];
    
    for st=1:sn
        
        % move the point mass towards xf, but not too far.
        while new_xt > xf + 1e-10 || abs(xf-new_xt) >= abs(xf-xt) 
            [new_xt, ctrls] = run_net(W1,W2,W3,st,1,xt);
        end
        
        xt = new_xt;
        traj = [traj, xt];
        ctrls_used(st,find(ctrls)) = 1;
        
        % stop if last point in sequence or xf reached, reward
        if st == sn || abs(xt-xf) < 1e-10
            % reward
            jerk = norm(diff([x0 x0 x0 traj xt xt xt],3),Inf);
            jerk_ma = jma_coef * jerk_ma + (1-jma_coef)*jerk;
            jerk_reward = jerk;
            %if jerk < jerk_ma - .05
            %    jerk_reward = 1/(1+jerk);
            %else
            %    jerk_reward = -.1;
            %end
            
            %jerk_reward = max((jerk_ma - jerk)/jerk_ma,0);
            x_dist_reward = exp(c1(x_dist_ma)*abs(xf-xt));
            s_dist_reward = exp(c2(s_dist_ma)*abs(sf-st));
            target_reward = x_dist_reward*s_dist_reward;
            target_reward_ma = trma_coef * target_reward_ma + (1-trma_coef)*target_reward;
            if target_reward < target_reward_ma
                target_reward = 0;
            end
            reward = max(target_reward - jerk_reward,0);
            %reward = (1-target_reward_ma)*max(target_reward - target_reward_ma,0) + target_reward_ma*jerk_reward;
            %reward = target_reward;
            
            % adjust weights/probabilities
            W1 = W1 + learn_rate*reward*ctrls_used;
            W1(W1<0) = 0;
            W1 = W1 ./ repmat(sum(W1,2),1,size(W1,2)); % normalize
            
            % track feedback parameters
            s_dist_ma = sma_coef*s_dist_ma + (1-sma_coef)*(abs(st-sf));
            x_dist_ma = xma_coef*x_dist_ma + (1-xma_coef)*(abs(xt-xf));
            
            % monitoring
            jerk_history = [jerk_history, jerk];
            s_dist_history = [s_dist_history, abs(st-sf)];
            x_dist_history = [x_dist_history, x_dist_ma];
            reward_history(:,t) = [reward;x_dist_reward;s_dist_reward;target_reward;jerk_reward];
            
            break
        end
        
    end
    
    % MONITORING
    if mod(t,tm) == 0
        
        pause(2)
        disp(sprintf('Iteration %d',t))
        
        % plot pdfs
        figure(1)
        for j=1:sn
            k = j;
            subplot(4,5,j)
            bar(1:nctrlrs,W1(k,:))
            title(sprintf('%d',k))
        end

        % plot jerk, sequence length, and distance histories and trajectory
        figure(2)
        subplot(4,1,1)
        plot_ma(s_dist_history, 20, t)
        title('Distance to sf (sequence length), MA')
        subplot(4,1,3)
        plot_ma(jerk_history, 20, t)
        hold on;
        plot(.1*ones(1,t),'r')
        hold off;
        title('Jerk MA') 
        subplot(4,1,2)
        plot_ma(x_dist_history, 20, t)
        title('Distance to xf MA')
        subplot(4,1,4)
        plot(0:size(traj,2),[0,traj])
        title('Sample Trajectory')

        
        % reward function plots
        figure(3)
        subplot(7,2,[1,2])
        plot_ma(reward_history(1,:),20,t)
        title('Reward History MA')
        subplot(7,2,[3,4])
        plot_ma(reward_history(2,:),20,t)
        title('xf reward history MA')
        subplot(7,2,[5,6])
        plot_ma(reward_history(3,:),20,t)
        title('sf reward history MA')
        subplot(7,2,[7,8])
        plot_ma(reward_history(4,:),20,t)
        title('target reward history MA')
        subplot(7,2,[9,10])
        plot_ma(reward_history(5,:),20,t)
        title('jerk reward history MA')
        subplot(7,2,11)
        x_dist = 0:g:xf;
        plot(x_dist,c1(x_dist));
        hold on;
        plot(x_dist_ma,c1(x_dist_ma),'o');
        hold off;
        title('xf reward decay coef')
        subplot(7,2,12)
        plot(x_dist,exp(c1(x_dist_ma)*x_dist))
        title('Current xf reward function')
        subplot(7,2,13)
        plot(1:sf,c2(1:sf))
        hold on
        plot(s_dist_ma,c2(s_dist_ma),'o')
        hold off
        title('sf reward decay coef')
        subplot(7,2,14)
        s_dist = 0:.1:sf;
        plot(s_dist,exp(c2(s_dist_ma)*s_dist))
        title('Current sf reward function')
        
    end
end
