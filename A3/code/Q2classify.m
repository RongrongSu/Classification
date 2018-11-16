%classifer
function[yhat]=classify(Xtest,b,b0)
[d,n] = size(X);
yhat=(b'*Xtest+b0)';
for i=1:n
    if yhat(i)>=0 
        yhat(i)=1;
    else
        yhat(i)=-1;
    end
end
end
