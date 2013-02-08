classdef StateMonitor < handle
    %MONITOR This is used for monitoring values over time
    
    properties
        history
        dim
        max_t
    end
    
    methods
        function self = StateMonitor(state0, max_t)
            self.dim = size(state0,2);
            self.max_t = max_t;
            self.history = zeros(self.dim,max_t+1);
            self.history(:,1) = state0;
        end
        
        function record(self, state, t)
            self.history(:,t+1) = state;
        end
        
        function plot(self, fig_num)
            n = ceil(self.dim^.5);
            figure(fig_num)
            for i=1:self.dim
                subplot(n,n,i)
                plot(self.history(i,:));
            end
        end
        
    end
    
end

