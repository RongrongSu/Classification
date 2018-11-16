rng(1);
num=5;
[idx, mu]=kmeans(X', num);
mu=mu';
sigma=zeros(num, 1);
phi=zeros(num,size(X,2));
for i=1:size(X,2)
sigma(idx(i), 1)=sigma(idx(i), 1)+norm(X(:,i)-mu(:,idx(i)))^2;
end

for i=1:num
    sigma(i,1)=sigma(i,1)/sum(idx==i);
end

for i=1:num
    for j=1:size(X,2)
        phi(i,j)=exp(-norm(X(:,j)-mu(:,i))^2/(sigma(i,1)+0.000001));
    end    
end   


W=inv(phi*phi')*phi*y;
y_est=W'*phi;
for i=1:length(y_est)
    if y_est(i)>=0.5
        y_est(i)=1;
    else
        y_est(i)=0;
    end
end

error_rate = 1-sum(y_est'==y)/length(y);


