clear;
clc;
close all;
data=xlsread('energy_efficiency_data.csv');
tic;
% one hot
ind=data(:,6);
vec = ind2vec(ind');
[ind,n] = vec2ind(vec);
vec2 = full(ind2vec(ind,n))';
ind=data(:,8)+ones(768,1);
vec = ind2vec(ind');
[ind,n] = vec2ind(vec);
vec3 = full(ind2vec(ind,n))';
data_onehot=[data(:,1:5) vec2(:,2:5) data(:,7) vec3 data(:,9)];



num_feature=16;
num_layer = 4;
num_train=576;
N =18;     % Number of samples in each mini-batch
num_minibatch = num_train / N;   % Number of mini-batches
num_test=192;

learning_rate =0.0000000022;
bias_rate=0.000000001;
time=30000;
D(1) = 15;
D(2) = 10;
D(3) = 10;
D(4) = 1;


train_predict=zeros(num_train,1);
test_predict =zeros(num_test,1);
train_E=zeros(time,1);
test_E=zeros(time,1);
record=zeros(1,9);
% affect 
data_affect=cell(1, 9);

for i=1:9
    if i==6
        A=zeros(768,4);
        data_affect{i}=[data_onehot(:,1:5),A,data_onehot(:,10:end)];
    elseif i==8
        A=zeros(768,6);
        data_affect{i}=[data_onehot(:,1:10),A,data_onehot(:,17)];
    elseif i==9
         data_affect{i}=data_onehot;
    elseif i==7
        temp=data_onehot;
        temp(:,10)=0;
        data_affect{i}=temp;
    else
        temp=data_onehot;
        temp(:,i)=0;
        data_affect{i}=temp;
    end
end

for K=1:9

    w = cell(1, num_layer);  % weights
    w{1}(1:num_feature, 1:D(1)) = randn(num_feature, D(1))/sqrt(num_feature/2);
    for i = 1:(num_layer-1)
    w{i+1}(1:D(i), 1:D(i+1)) = randn(D(i), D(i+1))/sqrt(D(i)/2);
    end
    grad_map = cell(1, num_layer);  % Initialize gradients
    grad_map{1}(1:num_feature, 1:D(1)) = 0;
    for i = 1:(num_layer-1)
    grad_map{i+1}(1:D(i), 1:D(i+1)) = 0;
    end
    delta = cell(1, num_layer);  % Initialize deltas
    for i = 1:num_layer
    delta{i}(num_train, D(i)) = 0;
    end
    bias = cell(1, num_layer);   % Initialize bias
    for i = 1:num_layer
    bias{i}(1:D(i)) = 0.1;
    end
   out = cell(1, num_layer);
    for i = 1:num_layer
    out{i}(:, 1:D(i)) = 0;
    end
   a = cell(1, num_layer);
   for i = 1:num_layer
    a{i}(:, 1:D(i)) = 0;
   end
for iteration = 1:time
    data_now=data_affect{K};
    % shuffle
    order=randperm(576);
    order2=randperm(192);
    data_one_train =  data_now(1:576,:);
    data_one_test=  data_now(577:768,:);
    data_suffle_train=data_one_train(order,:);
    data_suffle_test=data_one_test(order2,:);

   [A,order]=sort(order);
   [A,order2]=sort(order2);


   % depart train and test set
    data_train=data_suffle_train(:,1:16);
    y_train=data_suffle_train(:,17);
    data_test=data_suffle_test(:,1:16);
    y_test=data_suffle_test(:,17);
    %   Feed Forward 
  
        for minibatch = 1:num_minibatch
            train_x_mini = data_train(1+N*(minibatch-1):N+N*(minibatch-1), :);
            train_label_mini = y_train(1+N*(minibatch-1):N+N*(minibatch-1), :);
            for layer = 1:num_layer
                if layer == 1
                    input = train_x_mini;
                else
                    input = out{layer-1};
                end
                out{layer} = input*w{layer}+bias{layer};
                if layer ~= num_layer
                    out{layer}( out{layer}<0) = 0;  % ReLu function
                    a{layer}= out{layer};
                    a{layer}( a{layer}>0)=1;
                else     
                    out{layer} = out{layer} ;
                end
            end
            train_predict(1+N*(minibatch-1):N+N*(minibatch-1), :)  = out{layer};
    %   Back Propagation
        for layer = num_layer:-1:1
            if layer == num_layer
                delta{layer} = 2.*(out{layer} - train_label_mini);
                grad_map{layer} = out{layer-1}'*delta{layer};
             
            elseif layer ~= 1
                delta_temp = delta{layer+1}*w{layer+1}';
                delta{layer} =  a{layer} .* delta_temp;
                grad_map{layer} = out{layer-1}'*delta{layer};
            else
                delta_temp = delta{layer+1}*w{layer+1}';
                delta{layer} =  a{layer} .* delta_temp;
                grad_map{layer} = train_x_mini'*delta{layer};
            end
        end
            for layer = 1:num_layer
                w{layer} = w{layer} - learning_rate.*grad_map{layer};
                bias{layer} =  bias{layer}-bias_rate.*sum(delta{layer});
            end
        end
        train_E(iteration)=sum((y_train-train_predict).^2);
        train_RMS(iteration)=(train_E(iteration)/num_train)^0.5;
     %     Verify Result
        for batch = 1:num_test
            for layer = 1:num_layer
                if layer == 1
                    input = data_test(batch,:);
                else
                    input = out{layer-1};
                end
                    out{layer} = input*w{layer}+bias{layer};
                if layer ~= num_layer
                    out{layer}( out{layer}<0) = 0;  % ReLu function
                else
                    out{layer} = out{layer} ;
                end
            end
            test_predict(batch) = out{layer};
        end
        test_E(iteration)=sum((y_test-test_predict).^2);
        test_RMS(iteration)=(test_E(iteration)/num_test)^0.5;
    
   
end

record(1,K)=train_RMS(end);
record(2,K)=test_RMS(end);

figure,
plot(1:time, train_E);
title('train curve');
ylabel('square loss');
xlabel('#of epoch');
figure,
plot(1:time, test_E);
title('test curve');
ylabel('square loss');
xlabel('#of epoch');

figure,
plot(1:576,y_train(order));
title('heat load for training dataset');
ylabel('heat load');
xlabel('#th case');
hold on;
plot(1:576,train_predict(order));
legend('label','predict');
hold off;
figure,
plot(1:192,y_test(order2));
title('heat load for test dataset');
ylabel('heat load');
xlabel('#th case');
hold on;
plot(1:192,test_predict(order2));
legend('label','predict');
hold off;

end
toc;


