
size1=size(Xtrain,1);
size2=size(Xtrain,2);
nodes = 10;
epoch = 100;
I_matrix = -1*ones(size1,nodes);
u1 = I_matrix+rand(size1,nodes)*2; %33*5
u2 = -1*ones(nodes,1)+rand(nodes,1)*2; %5*1
layer1_z = zeros(nodes,size2); %5*176
layer1_a = zeros(nodes,size2); %5*176
y_input = zeros(1,size2); %1*176
y_output = zeros(1,size2);
lr=0.01; %learning rate
decay=0; %weight decay
delta_u2=zeros(nodes,1);
p1=zeros(nodes,1);
p2=zeros(nodes,1);
delta_u1=zeros(size1,nodes);
train_error1=zeros(nodes,1);
test_error1=zeros(nodes,1);

for n=1:nodes
    for e=1:epoch
         for i = 1:size2;
            layer1_a(:,i)=u1'*Xtrain(:,i);%5*1
            layer1_z(:,i)=1/(1+exp((layer1_a(:,i))*(-1)));
            y_input(1,i)=u2'*layer1_z(:,i);


            %from output to layer1
            delta_u2=-2*(ytrain(i,1)-y_input(1,i));%1

            %from layer1 to input
            delta_u1=delta_u2*Xtrain(:,i)*(u2.*layer1_z(:,i).*(ones(nodes,1)-layer1_z(:,i)))';%33*5


            u2=u2-(lr*delta_u2)-(u2*lr*decay);
            u1=u1-(lr*delta_u1)-(u1*lr*decay);  
         end
    end 
    %test train matrix
    train_class=zeros(size(Xtrain,2),1);
    ytrain_output=zeros(1,size(Xtrain,2));
    for i=1:size(Xtrain,2)
        layer1_a(:,i)=u1'*Xtrain(:,i);
        layer1_z(:,i)=1/(1+exp((layer1_a(:,i))*(-1)));
        ytrain_output(1,i)=u2'*layer1_z(:,i);
        if ytrain_output(1,i)>=0.5
          train_class(i,1)=1;
        else
          train_class(i,1)=0;
        end
    end

    number1=sum(train_class==ytrain);
    train_error11(n,1)=1-(number1/size(ytrain,1));


    %test test matrix
    test_class2=zeros(size(Xtest,2),1);
    ytest_input=zeros(1,size(Xtest,2));
    ytest_output=zeros(1,size(Xtest,2));
    for i=1:size(Xtest,2)
        layer1_a(:,i)=u1'*Xtest(:,i);
        layer1_z(:,i)=1/(1+exp((layer1_a(:,i))*(-1)));
        ytest_output(1,i)=u2'*layer1_z(:,i);
        if ytest_output(1,i)>=0.5;
          test_class2(i,1)=1;
        else
          test_class2(i,1)=0;
        end
    end

    number2=sum(test_class2==ytest);
    test_error11(n,1)=1-(number1/size(ytest,1));


end

plot(1:nodes,train_error11,'r');
hold on;
plot(1:nodes,test_error11,'g');
