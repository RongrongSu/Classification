
function[stump,err,s]=stump(x,y,w)
d=size(x,1);
n=size(x,2);
yhat=zeros(d,2);
vector=zeros(1,d);
err1=zeros(1,d);
err2=zeros(n,1);
for i=1:d
    for j=1:n
        if abs(x(j,:)')>=abs(x(j,j))
            isgreater=Ture;
            yhat(isgreater)=1;
        else
            isgreater=False;
            yhat(isgreater)=-1;
        end
        ind=(y~=yhat);
        error1(i) = w*ind/sum(w);
    end
    [error2(j),number] = min(error_row);
    vector(j) = x(j,number);
end

train_x=randn(2000,10);
test_x=randn (10000,10);
[train_m,train_n]=size(train_x);
[test_m,test_n]=size(test_x);
train_y=zeros(train_m,1);
for i=1:train_m
    temp=0;
        for j=1:n
            temp=train_x(i,j).^2;
        end
        if temp>9.34
            train_y(i)=1;
        else
            train_y(i)=-1;
        end
end

train_cot=0
for i=1:train_m
    if train_y(i)=1
        train_cot=train_cot+1;
    end
end

test_y=zeros(test_m,1);
for i=1:test_m
    temp=0;
        for j=1:n
            temp=test_x(i,j).^2;
        end
        if temp>9.34
            test_y(i)=1;
        else
            test_y(i)=-1;
        end
end

test_cot=0;
for i=1:test_m
    if test_y(i)=1
        test_cot=test_cot+1;
    end
end

weight=zeros(train_m,1);
for i=1:train_m
    weight(i)=1/train_m
end

it=200;
alpha=zeros(1,it);
train_error=zeros(it,1);
test_error=zeros(it,1);

for i=1:it
    [l,s]=findstp(train_x,train_y,weight)
    [row,~] = find(train_x == s);
    alpha = log((1-l)/(l+0.00001));
    train_hat=zeros(train_m,1);
    for j=1:train_m
        if train_x(row,j)>=s
            train_hat(j)=1;
        else
            train_hat(j)=-1;
        end
    end
    for j=1:test_m
        if test_x(row,j)>=s
            test_hat(j)=1;
        else
            test_hat(j)=-1;
        end
    end
    
    weight = weight .* (exp(alpha * (train_hat~=train_y)))';
    weight = weight ./ sum(weight);
    
    train_est(i,:)=(alpha*train_hat)';
    test_est(i,:)=(alpha*test_hat)';

    
    train_h=sum(train_est,1);
    if train_h>=0
        train_h=1;
    else
        train_h=-1;

    test_h=sum(test_est,1);
    if test_h>=0
        test_h=1;
    else
        test_h=-1;
    

    train_error(i)=sum(train_h'~=train_y)/train_m;
    test_error(i)=sum(test_h'~=test_y)/test_m;
    
end

end

plot(train_error)
hold on;
plot(test_error)
hold off;





    