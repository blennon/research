clear all

T = 12;
xf = 10;
x0 = 0;
J = 32*(xf-x0)/(T^3);
ctrls = [ones(1,T/4),2*ones(1,T/2),ones(1,T/4)];
states = zeros(4,size(ctrls,2)+1);
net = Net([0 0 0 0], [J -J]);
i = 2;
for c=ctrls
    states(:,i) = net.run(c);
    i = i + 1;
end

    