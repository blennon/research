% This is a simple neural network where the correct sequence of controllers
% is to be learned in order to move a point mass from x0 to xf

% Learns the Q function through TD and explores using epsilon-greedy

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
W2 = [transpose(0:g:w2_max),zeros(size(0:g:w2_max,2),1)];
W3 = [1,1,-1]';

% Q-function
Q = zeros(sn,nctrlrs);


%% Train the network

iters = 10000;   % Number of training iterations
tm = 500;       % Plot monitor every tm iterations
x0 = 0.0;       % starting location of point mass
xf = 5.4;       % final location of point mass
sf = 10;        % desired final sequence point
discount = .95; % amount to discount the rewards (SARSA)
epsilon = .01;  % epsilon greedy parameter

% LEARNING RATES
learn_rate = .1;
sig = @(x,ymax,ymin,xmin,xmax,beta) (ymax-ymin)./(1+exp(-(x - (xmax-xmin)/2)/beta)) + ymin;
% distance error coefficients
c1_start = log(.1)/(.9*xf); % coeff for .1 reward when .9*xf from xf
c1_end = log(.1)/(.1*xf);   % coeff for .1 reward when .1*xf from xf
c1_end = c1_start;
c1 = @(xdavg) sig(xdavg,c1_start,c1_end,0,xf,xf/8);
% sequence length error coefficients
c2_start = log(.1)/(.9*sf); % coeff for .1 reward when .9*sf from sf
c2_end = log(.1)/(.1*sf);   % coeff for .1 reward when .1*sf from sf
c2_end = c2_start;
c2 = @(sdavg) sig(sdavg,c2_start,c2_end,0,sf,sf/8);
% jerk error coefficients
cj_start = log(.1)/(.9);
cj_end = log(.1)/.1;
cj_end = cj_start;
cj = @(jdavg) sig(jdavg,cj_start,cj_end,0,1,.1);

% moving average parameters
sma_coef = .95;     % s_dist moving average coefficient
xma_coef = .95;     % x_dist moving average coefficient
s_dist_ma = sf;     % s_dist MA starting value
x_dist_ma = xf;     % x_dist MA starting value
target_reward_ma = 0;
trma_coef = .9;
jerk_ma = 0;
jma_coef = .9;
j_dist_ma = 0;
xdr_ma = 0;
sdr_ma = 0;
jr_ma = 0;

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
    
    learn_rate = 1/(t^.6);
    epsilon = 1/(t^.65);
    
    xt = 0;
    new_xt = 0;
    ctrls_used = zeros(sn,nctrlrs);
    traj = [];
    
    for st=1:sn
        

        % move the point mass towards xf, but not too far.
        while new_xt > xf + 1e-10 || abs(xf-new_xt) >= abs(xf-xt) 
            [new_xt, ctrls] = run_net_epsilon(Q,W2,W3,st,xt,epsilon);
        end
        
        xt = new_xt;
        traj = [traj, xt];
        ctrls_used(st,find(ctrls)) = 1;
                
        % stop if last point in sequence or xf reached, reward
        if st == sf || abs(xt-xf) < 1e-10
            % jerk
            jerk = norm(diff([x0 x0 x0 traj xt xt xt],3),Inf);
            j_dist_ma = jma_coef*j_dist_ma + (1-jma_coef)*abs(jerk-.1);
            jerk_ma = jma_coef * jerk_ma + (1-jma_coef)*jerk;
            
            % reward
            jerk_reward = exp(cj(j_dist_ma)*abs(jerk-.1));
            jr_ma = trma_coef* jr_ma + (1-trma_coef)*jerk_reward;
            x_dist_reward = exp(c1(x_dist_ma)*abs(xf-xt));
            s_dist_reward = exp(c2(s_dist_ma)*abs(sf-st));
            xdr_ma = trma_coef* xdr_ma + (1-trma_coef)*x_dist_reward;
            sdr_ma = trma_coef* sdr_ma + (1-trma_coef)*s_dist_reward;
            target_reward = x_dist_reward*s_dist_reward;
            target_reward_ma = trma_coef * target_reward_ma + (1-trma_coef)*target_reward;
            %reward = x_dist_reward - xdr_ma + s_dist_reward - sdr_ma;
            
            %reward = target_reward *jerk_reward;
            reward = target_reward - target_reward_ma + target_reward_ma*(jerk_reward-jr_ma);
            
            % adjust weights
            Q = adjust_Q(Q, reward, ctrls_used, discount, learn_rate, st);

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
        
        %pause(2)
        disp(sprintf('Iteration %d',t))
        
        % plot pdfs
        figure(1)
        for j=1:sn
            k = j;
            subplot(4,5,j)
            bar(1:nctrlrs,Q(k,:))
            title(sprintf('%d',k))
        end

        % plot jerk, sequence length, and distance histories and trajectory
        figure(2)
        subplot(3,1,1)
        plot_ma(s_dist_history, 20, t)
        title('Distance to sf (sequence length), MA')
        subplot(3,1,3)
        plot_ma(jerk_history, 20, t)
        hold on;
        plot(.1*ones(1,t),'r')
        hold off;
        title('Jerk MA') 
        subplot(3,1,2)
        plot_ma(x_dist_history, 20, t)
        title('Distance to xf MA')
        
        % reward function plots
        figure(3)
        subplot(8,2,[1,2])
        plot_ma(reward_history(1,:),20,t)
        title('Reward History MA')
        subplot(8,2,[3,4])
        plot_ma(reward_history(2,:),20,t)
        title('xf reward history MA')
        subplot(8,2,[5,6])
        plot_ma(reward_history(3,:),20,t)
        title('sf reward history MA')
        subplot(8,2,[7,8])
        plot_ma(reward_history(4,:),20,t)
        title('target reward history MA')
        subplot(8,2,[9,10])
        plot_ma(reward_history(5,:),20,t)
        title('jerk reward history MA')
        subplot(8,2,11)
        x_dist = 0:g:xf;
        plot(x_dist,c1(x_dist));
        hold on;
        plot(x_dist_ma,c1(x_dist_ma),'o');
        hold off;
        title('xf reward decay coef')
        subplot(8,2,12)
        plot(x_dist,exp(c1(x_dist_ma)*x_dist))
        title('Current xf reward function')
        subplot(8,2,13)
        plot(1:sf,c2(1:sf))
        hold on
        plot(s_dist_ma,c2(s_dist_ma),'o')
        hold off
        title('sf reward decay coef')
        subplot(8,2,14)
        s_dist = 0:.1:sf;
        plot(s_dist,exp(c2(s_dist_ma)*s_dist))
        title('Current sf reward function')
        subplot(8,2,15)
        x = 0:.05:2;
        plot(x,cj(x))
        hold on
        plot(j_dist_ma,cj(j_dist_ma),'o')
        hold off
        title('jerk reward decay coef')
        subplot(8,2,16)
        j_dist = 0:.01:2;
        plot(j_dist,exp(cj(j_dist_ma)*j_dist))
        title('Current jerk reward function')
        
        figure(4)
        % optimal
        T=12;
        J = 32*(xf-x0)/(T^3);
        jerk = [J*ones(1,T/4),-J*ones(1,T/2),J*ones(1,T/4)];
        accel = cumsum(jerk);
        veloc = cumsum(accel);
        pos = [x0 cumsum(veloc)];

        % learned
        l = [x0 traj xt xt];
        subplot(2,2,1)
        plot(0:size(l,2)-1,l)
        hold on;
        plot(0:size(pos,2)-1,pos,'r')
        hold off;
        title('Position')
        for i=1:3
            subplot(2,2,i+1)
            li = diff([x0*ones(1,i) l],i);
            pi = diff([x0*ones(1,i) pos],i);
            plot(0:size(li,2)-1,li)
            hold on;
            plot(0:size(pi,2)-1,pi,'r')
            hold off;
            title(sprintf('D%d',i))
        end

        
    end
end
