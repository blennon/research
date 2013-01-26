dt1 = [.1 .2 .3 .4 .5 .5 .4 .3 .2 .1];
dt2 = .3 * ones(1,10);

t1 = [0 0 0 cumsum(dt1) 3 3 3];
t2 = [0 0 0 cumsum(dt2) 3 3 3];

norm(diff([t1],3),Inf)
norm(diff([t1],3),2)
norm(diff([t2],3),Inf)
norm(diff([t2],3),2)

close all;
figure(1)
subplot(2,2,1)
plot(t1)
for i=1:3
    subplot(2,2,i+1)
    plot(diff(t1,i))
end