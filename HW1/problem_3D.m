clear;
clc;
close all;
load('spam_data.mat');

num_train = 2000;
num_test = 500;
N = 100;  % Number of samples in each mini-batch
num_minibatch = num_train / N;  % Number of mini-batches
num_feature = 40;
learning_rate = 0.00001;
bias_rate=0.0000002
K = 2;  % Number of classes
num_layer = 5;  % Number of layers
D(1) = 150; % Nodes in first layer
D(2) = 100;
D(3) = 30;
node=3;
D(4) = node;
D(5) = K;

w = cell(1, num_layer);  % Initailize weights
w{1}(1:num_feature, 1:D(1)) =  normrnd(0, 0.2, num_feature, D(1));
for i = 1:(num_layer-1)
    w{i+1}(1:D(i), 1:D(i+1)) = normrnd(0, 0.2, D(i), D(i+1));
end
grad = cell(1, num_layer);  % Initialize gradients
grad{1}(1:num_feature, 1:D(1)) = 0;
for i = 1:(num_layer-1)
    grad{i+1}(1:D(i), 1:D(i+1)) = 0;
end
delta = cell(1, num_layer);  % Initialize deltas
for i = 1:num_layer
    delta{i}(N, D(i)) = 0;
end
bias = cell(1, num_layer);   % Initialize bias
for i = 1:num_layer
    bias{i}(1:D(i)) = 0.1;
end

actvivate = cell(1, num_layer);
for i = 1:num_layer
    actvivate{i}(:, 1:D(i)) = 0;
end
h = cell(1, num_layer);
for i = 1:num_layer
    h{i}(:, 1:D(i)) = 0;
end
y = zeros(num_train, K);
feature=zeros(40, node);
feature2=zeros(20, node);

time=500;
acc_train=zeros(1,time);
for iteration = 1:time
    iteration
    for minibatch = 1:num_minibatch
        train_x_mini = train_x(1+N*(minibatch-1):N+N*(minibatch-1), :);
        train_label_mini = train_y(1+N*(minibatch-1):N+N*(minibatch-1), :);
        for layer = 1:num_layer
            if layer == 1
                input = train_x_mini;
            else
                input = h{layer-1};
            end
            actvivate{layer} = input*w{layer}+bias{layer};
            if layer ~= num_layer
                h{layer} = actvivate{layer};
                h{layer}(h{layer}<0) = 0;  % ReLu function
                actvivate{layer}= h{layer};
                 actvivate{layer}( actvivate{layer}>0)=1;
            else
                temp = softmax(actvivate{layer}'); % Softmax function
                h{layer} = temp';
            end
        end
        y(1+N*(minibatch-1):N+N*(minibatch-1), :) = h{layer};
        label(1+N*(minibatch-1):N+N*(minibatch-1), :) = h{layer-1};
        feature(minibatch, :)=w{layer}(:,1);
        feature2(minibatch, :)=w{layer}(:,2);
      
        %    Back Propagation   
        for layer = num_layer:-1:1
            if layer == num_layer
                delta{layer} = (h{layer} - train_label_mini);
                grad{layer} = h{layer-1}'*delta{layer};
            elseif layer ~= 1
                delta_temp = delta{layer+1}*w{layer+1}';
               delta{layer} = 1 * (actvivate{layer} .* delta_temp);
                grad{layer} = h{layer-1}'*delta{layer};
            else
                delta_temp = delta{layer+1}*w{layer+1}';
               delta{layer} = 1 * (actvivate{layer} .* delta_temp);
                grad{layer} = train_x_mini'*delta{layer};
            end
        end
        
        for layer = 1:num_layer
            w{layer} = w{layer} - learning_rate.*grad{layer};
            bias{layer} =  bias{layer}-bias_rate.*sum(delta{layer});
        end
    end
   
    E_cross(iteration)=-trace(train_y*log(y)')/2000;
    [value, train_result] = max(y');
    label=label-mean(label);
    [value, train_answer] = max(train_y');
    num_right = sum(train_result == train_answer);
    acc_train(iteration) = ((num_train-num_right)/num_train);
    if iteration==3
         ind=find(train_answer==1);
         ind2=find(train_answer==2);
         X2=label(ind,:);
         X3=label(ind2,:);
          s=2*ones(size(X2,1),1);
         s1=2*ones(size(X3,1),1);
         figure,
         scatter3(X2(:,1),X2(:,2),X2(:,3),s,'r');
         hold on;
         scatter3(X3(:,1),X3(:,2),X3(:,3),s1,'b');
         title('3D feature 3 epoch');
         legend('class 1','class 2');
        
    elseif iteration==300  
         ind=find(train_answer==1);
         ind2=find(train_answer==2);
         X2=label(ind,:);
         X3=label(ind2,:);
          s=2*ones(size(X2,1),1);
         s1=2*ones(size(X3,1),1);
         figure,
         scatter3(X2(:,1),X2(:,2),X2(:,3),s,'r');
         hold on;
         scatter3(X3(:,1),X3(:,2),X3(:,3),s1,'b');
         title('3D feature 300 epoch');
         legend('class 1','class 2');
    end
      %Verify Result
    for layer = 1:num_layer
        if layer == 1
            input = test_x;
        else
            input = h{layer-1};
        end
        actvivate{layer} = input*w{layer};
        if layer ~= num_layer
            h{layer} = actvivate{layer};
            h{layer}(h{layer}<0) = 0;  % ReLu function
        else
            temp = softmax(actvivate{layer}'); % Softmax function
            h{layer} = temp';
        end
    end
    [value, test_result] = max(h{num_layer}');
    [value, test_answer] = max(test_y');
    num_right = sum(test_result == test_answer);
    error_test(iteration) = ((num_test-num_right)/num_test) ;
    
end

figure,
plot(1:time, error_test);
title('test error rate');
ylabel('Average cross entropy');
xlabel('Number of epochs');
figure,
plot(1:time, acc_train);
title('train error rate');
ylabel('Average cross entropy');
xlabel('Number of epochs');

figure,
plot(1:time,E_cross);
title('training loss');
ylabel('Average cross entropy');
xlabel('Number of epochs');
