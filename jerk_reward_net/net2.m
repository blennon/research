% This is a simple neural network where the correct sequence of controllers
% is to be learned in order to move a point mass from x0 to xf

% RESULTS: This network learns to move the mass from x0 to xf in a straight
% line in desired number of steps.

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

iters = 2000;   % Number of training iterations
tm = 400;       % Plot monitor every tm iterations
x0 = 0.0;    % starting location of point mass
xf = 2.0;    % final location of point mass
sf = 10;     % desired final sequence point
ma_coef = .1; % moving average coefficient for computing jerk MA

% learning rates
learn_rate = .05;
% distance error coefficient
c1_start = log(.1)/(.8*xf); % coeff for .1 reward when .9*xf from xf
c1_end = log(.1)/(.1*xf);   % coeff for .1 reward when .1*xf from xf
% sequence length error coefficient
c2_start = log(.1)/(.8*sf); % coeff for .1 reward when .9*sf from sf
c2_end = log(.1)/(.1*sf);   % coeff for .1 reward when .1*sf from sf
c_decay = -2/iters*log(1/3); % decayed to 1/3 at iters/2
c1 = @(t) (c1_start - c1_end)*exp(-2*c_decay*t) + c1_end;
c2 = @(t) (c2_start - c2_end)*exp(-0.2*c_decay*t) + c2_end;

% IMPLICIT VARIABLES LEGEND
% si - current sequence step number
% xi - current value of x for step i
% t - training iteration

% monitoring
jerk_history = [];
si_history = [];
reward_history = [];
si_ma = 0;

for t=1:iters

    disp(t)
    
    xi = 0;
    new_xi = 0;
    ctrls_used = zeros(sn,nctrlrs);
    traj = [];
    jerk_ma = 0;
    
    for si=1:sn
        
        % move the point mass towards xf, but not too far.
        while new_xi > xf + 1e-10 || abs(xf-new_xi) >= abs(xf-xi) 
            [new_xi, ctrls] = run_net(W1,W2,W3,si,1,xi);
        end
        
        xi = new_xi;
        traj = [traj, xi];
        ctrls_used(si,find(ctrls)) = 1;
        
        % stop if last point in sequence or xf reached, reward
        if si == sn || abs(xi-xf) < 1e-10
            % reward
            jerk = norm(diff([0,0,0,traj,0,0,0],3));
            jerk_ma = ma_coef*jerk + (1-ma_coef)*jerk_ma;
            reward = exp(c1(t)*xf*abs(xf-xi))*exp(c2(t)*abs(sf-si)); %* min(0,(jerk_ma - jerk));
            
            % adjust weights/probabilities
            W1 = W1 + learn_rate*reward*ctrls_used;
            W1(W1<0) = 0;
            W1 = W1 ./ repmat(sum(W1,2),1,size(W1,2)); % normalize
            
            % monitoring
            jerk_history = [jerk_history, jerk_ma];
            si_ma = .9*si_ma + .1*si;
            si_history = [si_history, si_ma];
            reward_history = [reward_history, reward];
            
            break
        end
        
    end
    
    % MONITORING
    if mod(t,tm) == 0
        
        %pause(3)
        
        % plot pdfs
        f1 = figure(1);
        for j=1:sn
            k = j;
            subplot(4,5,j)
            bar(1:nctrlrs,W1(k,:))
            title(sprintf('%d',k))
        end

        % plot jerk history and sample trajectory
        f2 = figure(2);
        subplot(4,1,1)
        plot(si_history)
        title('Sequence length to xf, MA')
        subplot(4,1,2)
        plot(jerk_history)
        title('Jerk MA') 
        subplot(4,1,3)
        plot(0:size(traj,2),[0,traj])
        title('Sample Trajectory')
        subplot(4,1,4)
        plot(reward_history)
        title('Reward History')
        
        % plot learning rate coefficients
        f3 = figure(3);
        subplot(2,2,1)
        plot(1:t,c1(1:t))
        title('xf reward decay coef')
        subplot(2,2,2)
        x_dist = 0:.1:xf;
        plot(x_dist,exp(c1(t)*x_dist))
        title('Current xf reward function')
        subplot(2,2,3)
        plot(1:t,c2(1:t))
        title('sf reward decay coef')
        subplot(2,2,4)
        s_dist = 0:.1:sf;
        plot(s_dist,exp(c2(t)*s_dist))
        title('Current sf reward function')
        
        set(f1,'Position',[1,1,1680,1050])
        set(f2,'Position',[1,1,1680,1050])
        set(f3,'Position',[1,1,1680,1050])
        saveas(f1,sprintf('f1_iter%d',t),'jpg');
        saveas(f2,sprintf('f2_iter%d',t),'jpg');
        saveas(f3,sprintf('f3_iter%d',t),'jpg');
    end
    
    
end
