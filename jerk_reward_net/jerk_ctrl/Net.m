classdef Net < handle
    %NET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state = [0 0 0 0] % x,v,a,j
        ctrlr_weights
        vmax
        amax
        beta
    end
    
    methods
        function self=Net(start_state, ctrlr_weights, vmax, amax, beta)
            % Network constructor
            self.state = start_state;
            self.ctrlr_weights = ctrlr_weights;
            self.vmax = vmax;
            self.amax = amax;
            self.beta = beta;
        end
        
        function state = run(self, ctrlr)
            % Run the network/system using ctrlr as the controller, update
            % the state.
            self.state(4) = self.ctrlr_weights(ctrlr);
            self.state(3) = self.sigmoid(self.state(3) + self.state(4),self.amax,-self.amax,self.beta);
            self.state(2) = self.sigmoid(self.state(2) + self.state(3),self.vmax,-self.vmax,self.beta);
            self.state(1) = self.state(1) + self.state(2);
            state = self.state;
        end
    end
    
    methods (Static)
        function y = sigmoid(x,ymax,ymin, beta)
            y = (ymax-ymin)./(1+exp(-(x)/beta)) + ymin;
        end    
    end
    
end

