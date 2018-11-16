%harmargin
function [b,b0]=hardmarg(X,y)
[d,n]=size(X);
Y=diag(y)
H=(X*Y)'*(X*Y);
f=-1*ones(n,1);
Aeq=Y';
beq=0;
lb=zeros(n,1);
alpha=quadprog(H,f,[],[],Aeq,beq,lb);
b=X*Y*alpha;
[argVal,argMax]=max(alpha);
b0=1/y(argMax,1) - b'* X(:, argMax);
end