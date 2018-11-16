%STANDARD CROSS VALIDATION
num=20; %cluster 1-20 centers
cv_num=10; %divide raw date into 13 groups 
a=35;
error_train=zeros(num,1);
error_test=zeros(num,1);
for n=2:num
    rng(150);
    [idx, mu]=kmeans(X', n);
    mu=mu';
    sigma=zeros(n, 1);
    train_error=zeros(cv_num,1);
    test_error=zeros(cv_num,1);
    
    
    %calculate sigma
    for l=1:size(X,2)
        sigma(idx(l), 1)=sigma(idx(l), 1)+norm(X(:,l)-mu(:,idx(l)))^2;
    end
    for i=1:n
        sigma(i,1)=sigma(i,1)/sum(idx==i);
    end
    
    %cross validation
    for i=1:cv_num
        
        %split data
        train=[X(:,1:a*(i-1)),X(:,(a*(i)+1):size(X,2))];
        test=X(:,(a*(i-1)+1):a*(i));
        y_train=[y(1:a*(i-1),1);y((a*(i)+1):size(X,2),1)];
        y_test=y((a*(i-1)+1):a*i);
       
        
        %calculate phi&W&train-error for train data
        phi=zeros(n,size(train,2));
        for k=1:n
            for j=1:size(train,2)
                phi(k,j)=exp(-norm(train(:,j)-mu(:,k))^2/(sigma(k,1)+0.000001));
            end    
        end
        W=pinv(phi*phi')*(phi*y_train);
        train_est=phi'*W;
        for t=1:length(train_est)
            if train_est(t)>=0.5
                train_est(t)=1;
            else
                train_est(t)=0;
            end
        end
        train_error(i,1)=1-sum(train_est==y_train)/size(y_train,1);
        
        %calculate phi&W&test-error for test data
        phi=zeros(n,a);
        for k=1:n
            for j=1:a
                phi(k,j)=exp(-norm(test(:,j)-mu(:,k))^2/(sigma(k,1)+0.000001));
            end    
        end   
        
        test_est=phi'*W;
        for w=1:length(test_est)
            if test_est(w)>=0.5
                test_est(w)=1;
            else
                test_est(w)=0;
            end
        end  
        test_error(i,1)=1-sum(test_est==y_test)/size(y_test,1);
    end
    error_train(n,1)=sum(train_error)/cv_num;
    error_test(n,1)=sum(test_error)/cv_num;
end
plot(2:num,error_train(2:num,1),'r');
hold on;
plot(2:num,error_test(2:num,1),'g');



