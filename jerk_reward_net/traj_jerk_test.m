close all;
clear all;
clc;


%% Jerk control policy
T = 12;
x0 = 0;
xf = 5.4;
J = 32*(xf-x0)/(T^3);
jerk = [J*ones(1,T/4),-J*ones(1,T/2),J*ones(1,T/4)];

figure(1)
subplot(2,2,4)
plot(jerk)
subplot(2,2,3)
accel = cumsum(jerk);
plot(accel)
subplot(2,2,2)
veloc = cumsum(accel);
plot(veloc)
subplot(2,2,1)
pos = cumsum(veloc);
plot(pos)

norm(diff([x0 x0 x0 pos xf xf xf],3),Inf)
