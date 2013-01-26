clear all;
close all;
clc;

% Instantiate net

% PARAMETERS
sn = 20; % max number of sequence points
g = .1; % granularity of spacing between W2 weights (i.e. granularity of control of alpha motor neurons)
alpha_max = 1; % maximum output value of alpha motor neurons
w2_max = 1; % maximum weight value impinging on the alpha MNs
nctrlrs = size(0:g:w2_max,2)^2; % number of controllers affecting the alpha MNs
nctrlrs = 2*nctrlrs^.5 - 1;
nctrlrs = size(0:g:w2_max,2);

% WEIGHTS -- (source x target)
W1 = 1/nctrlrs * ones(sn,nctrlrs);
%[X,Y] = meshgrid(0:g:w2_max,0:g:w2_max);
%W2 = [X(:),Y(:)];

W2 = [transpose(0:g:w2_max),zeros(size(0:g:w2_max,2),1)];
%W2 = [W2;fliplr(W2(2:end,:))];
W3 = [1,1,-1]';


% Train
iters = 10000;
xfinal_end = 2; % could increase over iters
xfinal_start = .3;
sn_start = 4;
sn_end = sn;
sn_tol = 3;

xinit = 0.0;
epsilon_start = .1; % could decay over iters
nctrl_start = 1; % could decay over iters
learn_rate_start = .1; % could decay over iters


jerk_run = [];
ctrlrs_picked = [];
s_trouble = [];
traj_len = [];
jerk_ma = 0;

for i=1:iters
    x = xinit;
    new_x = -1;
    traj = [];
    ctrls = zeros(1,nctrlrs);
    ctrls_used = zeros(sn_end,nctrlrs);

    % Decaying values
%     nctrl_use = max(floor(nctrl_start * exp(-i/(iters/2))),1);
%     epsilon = max(epsilon_start * exp(-i/(iters/2)),.01);
%     learn_rate = max(learn_rate_start * exp(-i/(iters/2)),.1);
    xfinal = floor(min(xfinal_start * exp(i/(iters/2)),xfinal_end)*10)/10
    sn = ceil(min(sn_start * exp(i/(iters/2)),sn_end))
    
    
    nctrl_use = nctrl_start;
    epsilon = epsilon_start;
    learn_rate = learn_rate_start;
    %xfinal = xfinal_end;
    
    bad = false;
    i
    for s=1:sn
        % generate a new point in the trajectory thats closer to the target
        while abs(new_x - xfinal) >= abs(x - xfinal)
            [new_x, ctrls] = run_net(W1,W2,W3,s,nctrl_use,x);
            %if i > 400 && s > 10
            %    s_trouble = [s_trouble,new_x-xfinal];
            %end
        end

        x = new_x;
        traj = [traj,new_x]
        ctrls_used(s,find(ctrls)) = 1/nctrlrs;
        
        if abs(x-xfinal) <= epsilon && abs(s-sn) <= sn_tol
            jerk = norm(diff([0,0,0,traj,0,0,0],3));
            jerk_ma = .2*jerk + .8*jerk_ma;
            jerk_run = [jerk_run, jerk_ma];
            traj_len = [traj_len, size(traj,2)];
            
            if size(jerk_run,2) < 10;
                bad = true;
                break
            end
            
            reward = (jerk_ma - jerk)/jerk_ma;
            %reward = 1/(1+jerk);
            W1 = W1 + learn_rate*reward*ctrls_used;
            W1(W1<0) = 0;
            W1 = W1 ./ repmat(sum(W1,2),1,size(W1,2));

            break
        elseif x>=xfinal + epsilon
            bad = true;
            break
        elseif abs(x-xfinal) <= epsilon && abs(s-sn) >= sn_tol
            bad = true;
            break
        end
    end
    
    if mod(i,50) == 0 && bad == false
        disp(i)
        nctrl_use
        epsilon
        learn_rate
        xfinal
        xfinal
        sn
        % plot histograms
        figure(1)
        for j=1:sn
            subplot(4,5,j)
            bar(1:nctrlrs,W1(j,:))
        end
        figure(2)
        plot(jerk_run)
        
        figure(3)
        hold on
        plot([0,traj])
    end
end
% figure(3)
% hist(s_trouble)
% figure(4)
% for j=10:29
%     subplot(4,5,j-9)
%     bar(1:nctrlrs,W1(j,:))
% end