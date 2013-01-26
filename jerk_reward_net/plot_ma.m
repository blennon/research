function [ output_args ] = plot_ma( x, l, t )
%PLOT_MA Plot a moving average of values in x up to position t with a
% filter of length l
%   Detailed explanation goes here
l = min(t,l);
plot(filter(ones(l,1),l,x(1,1:t)));

end

