%softmargin
function [b,b0] = SoftMarg(X,y,gamma)
[d,n]=size(X);
Y=diag(y)
H=(X*Y)'*(X*Y);
f=-1*ones(n,1);
Aeq=Y';
beq=0;
lb=zeros(n,1);
ub=gamma*ones(n,1);
alpha = quadprog(H, f, [], [], A, bb, lb, ub);
b=X*Y*alpha;
while alpha(arg,1) < 0.01 || alpha(arg,1) > gamma
arg = arg+1;
end
b0s = 1/y(arg,1) - b'* X(:, arg);
end
