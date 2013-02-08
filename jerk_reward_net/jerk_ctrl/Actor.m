classdef Actor < handle
    %ACTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Q   % action-value function
        n_ctrls % number of controllers
        max_t   % max number of time steps
        episode_history % history of controllers used during an epsidode
    end
    
    methods
        function self=Actor(n_ctrls,max_t)
            % Actor constructor
            self.n_ctrls = n_ctrls;
            self.max_t = max_t;
            self.Q = zeros(max_t,n_ctrls);
            self.episode_history = zeros(max_t,1);
        end
        
        function c = choose_controller(self, t, beta)
            % choose a controller according to the softmax policy
            c = randsample(self.n_ctrls,1,true,softmax(beta*self.Q(t,:)));
            self.episode_history(t,1) = c;
        end
        
        function update_Q_sarsa(self, st, at, rt, st1, at1, alpha, gamma)
            % Update the Q function according to the SARSA rule
            if st == self.max_t
                self.Q(st,at) = self.Q(st,at) + alpha*(rt - self.Q(st,at));
            else
                self.Q(st,at) = self.Q(st,at) + alpha*(rt + gamma*self.Q(st1,at1) - self.Q(st,at));
            end
        end
        
        function update_Q_end(self, reward, alpha)
            % Update the action value function Q at the end of each episode
            % by moving the Q value for the controller used in each step of
            % the sequence towards the reward value.
            indcs = sub2ind(size(self.Q),1:self.max_t,self.episode_history');
            self.Q(indcs) = self.Q(indcs) + alpha*(reward*ones(self.max_t,1)' - self.Q(indcs));
        end
        
        function reset_history(self)
            self.episode_history = zeros(self.max_t,1);
        end
        
        % ---- PLOTTING ----
        function plot_Q(self, fig_num)
            % plot the action-values
            figure(fig_num)
            n = ceil(self.max_t^.5);
            for i=1:self.max_t
                subplot(n,n,i)
                bar(1:self.n_ctrls, self.Q(i,:))
                title(sprintf('%d',i))
            end
        end
        function plot_policy(self, beta, fig_num)
            % plot the policies taken for action values
            figure(fig_num)
            n = ceil(self.max_t^.5);
            for i=1:self.max_t
                subplot(n,n,i)
                bar(1:self.n_ctrls, softmax(beta*self.Q(i,:)'))
                title(sprintf('%d',i))
            end
        end
    end
    
end

