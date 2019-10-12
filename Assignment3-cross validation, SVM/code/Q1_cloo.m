%STANDARD CROSS VALIDATION without iteration
num=20; %cluster 1-20 centers
rng(150);
y_error=zeros(num/2,1);
for n=2:num
    %cluster data using Kmeans
    [idx,mu]=kmeans(X',n);
    mu=mu';
    sigma=zeros(n,1);
    for j=1:size(X,2)
        sigma(idx(j),1)=sigma(idx(j),1)+norm(X(:,j)-mu(:,idx(j)))^2;
    end
    for i=1:n
        sigma(i,1)=sigma(i,1)/sum(idx==i);
    end
    
    %calculate phi&W&train-error for train data
    phi=zeros(n,size(X,2));
    for i=1:n
        for j=1:size(X,2)
            phi(i,j)=exp(-norm(X(:,j)-mu(:,i))^2/(sigma(i,1)+0.0001));
        end
    end
    w=pinv(phi*phi')*(phi*y);
    y_est=phi'*w;
    H=phi'*pinv(phi*phi')*phi;
    for t=1:length(y_est)
        if y_est(t)>=0.5
            y_est(t)=1;
        else
            y_est(t)=0;
        end
    end
    y_error(i,1)=1-sum(y_est==y)/size(y,1);
end

plot(2:num,y_error(2:num,1),'g');

y_error(13,1)



