function [U1_final, A1_final, U2_final, A2_final, H_final, W1_final, W2_final, Q_final] = solveDLSH(X1, X2, Y, bits, K1, K2, param)
%% parameter setting
[row, col] = size(X1);
[~, colt] = size(X2);
H = sign(-1+(1-(-1))*rand(row, bits));
U1 = zeros(K1, col);
U2 = zeros(K2, colt);
A1 = ones(row, K1);
A2 = ones(row, K2);
W1 = rand(K1, bits);
W2 = rand(K2, bits);
Q = rand(10, bits);
threshold = 0.01;
lastF = 1000;
iter = 1;
maxIter = 20;
alpha = param.alpha;
betta1 = param.betta1;
betta2 = param.betta2;
sigma = param.sigma;
gamma = param.gamma;
%% random initialization
Q = (sigma*Y'*Y+gamma*eye(21)) \ (sigma*Y'*H);
A1 = (X1*U1'+betta1*H*W1') / (U1*U1'+betta1*W1*W1');
U1 = (A1'*A1+gamma*eye(K1)) \ (A1'*X1);
A2 = (alpha*X2*U2'+betta2*H*W2') / (alpha*U2*U2'+betta2*W2*W2');
U2 = (A2'*A2+gamma*eye(K2)) \ (A2'*X2);
W1 = (betta1*A1'*A1+gamma*eye(K1)) \ (betta1*A1'*H);
W2 = (betta2*A2'*A2+gamma*eye(K2)) \ (betta2*A2'*H);

%% offline training stage
while (iter<maxIter)
    %Update H
    H = sign(betta1*A1*W1+betta2*A2*W2+sigma*Y*Q);
    %Update Q
    Q = (sigma*Y'*Y+gamma*eye(21)) \ (sigma*Y'*H);
    %Update U1 A1
    A1 = (X1*U1'+betta1*H*W1') / (U1*U1'+betta1*W1*W1');
    U1 = (A1'*A1+gamma*eye(K1)) \ (A1'*X1);
    %Update U2 A2
    A2 = (alpha*X2*U2'+betta2*H*W2') / (alpha*U2*U2'+betta2*W2*W2');
    U2 = (A2'*A2+gamma*eye(K2)) \ (A2'*X2);
    %Update W
    W1 = (betta1*A1'*A1+gamma*eye(K1)) \ (betta1*A1'*H);
    W2 = (betta2*A2'*A2+gamma*eye(K2)) \ (betta2*A2'*H);
    
    % compute objective function
    norm1 = sum(sum((X1-A1*U1).^2));
    norm2 = sum(sum((X2-A2*U2).^2));
    norm3 = sum(sum((H-A1*W1).^2));
    norm4 = sum(sum((H-A2*W2).^2));
    norm5 = sum(sum((H-Y*Q).^2));
    norm6 = sum(sum((W1).^2)) + sum(sum((W2).^2)) + sum(sum((U1).^2)) + sum(sum(U2).^2) + sum(sum(Q).^2);
    currentF = norm1 + alpha*norm2 + betta1*norm3 + betta2*norm4 + sigma*norm5 + gamma*(norm6);
    %     fprintf('\ncurrentF at iteration %d: %.2f, obj: %.4f\n', iter, currentF, lastF - currentF);
    if ((lastF - currentF) < threshold)
        if iter > 1
            return;
        end
    end
    U1_final = U1;
    A1_final = A1;
    U2_final = U2;
    A2_final = A2;
    H_final = H;
    W1_final = W1;
    W2_final = W2;
    Q_final = Q;
    
    iter = iter + 1;
    lastF = currentF;
end
return;

