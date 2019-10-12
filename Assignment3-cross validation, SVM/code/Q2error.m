%error
count = 0;
for i = 1:size(Xtest, 2)
if yhat(i,1) ~= ytest(i,1)
count = count+1;
end
end
test_error = count/size(X, 2);