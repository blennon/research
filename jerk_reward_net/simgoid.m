sig = @(x,a,b,xmin,xmax,beta) (a-b)./(1+exp(-(x - (xmax-xmin)/2)/beta)) + b;
sf =100;
x = 0:1:sf;
sig = @(x,ymax,ymin,xmin,xmax,beta) (ymax-ymin)./(1+exp(-(x - (xmax-xmin)/2)/beta)) + ymin;
close all; figure(1); plot(x,sig(x,c2_start,c2_end,0,sf,sf/10)); hold on; plot(x,(c2_start - c2_end)/sf * x + c2_end); hold off;