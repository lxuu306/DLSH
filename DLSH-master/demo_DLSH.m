clear all;warning off; clc;
load mir5k.mat;
fprintf('MIRFlickr dataset loaded...\n');
%% parameter setting
run = 3;
map = zeros(run,2);
BITS = [16 32 64 128];
K1 = 32;
K2 = 32;
alpha = 1e-3;
betta1 = 1e-3;
betta2 = 1e-3;
sigma = 1e-1;
gamma = 1e1;

%% data prepare
Ntrain = size(I_tr,1);
n_anchors = 1000;
sample = randsample(Ntrain, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);
sigmaI = 80;
sigmaT = 1;
PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhtT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhtT = [PhtT, ones(Ntrain,1)];
Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
Pht_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
Pht_testT = [Pht_testT, ones(size(Pht_testT,1),1)];
Phi_trainI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
Phi_trainI = [Phi_trainI, ones(size(Phi_trainI,1),1)];
Pht_trainT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
Pht_trainT = [Pht_trainT, ones(size(Pht_trainT,1),1)];

%% offline training stage
for bi=1:4
    for i = 1 : run
        I_temp = PhiI;
        T_temp = PhtT;
        [row, col] = size(I_temp);
        [rowt, ~] = size(T_temp);
        param.alpha = alpha;
        param.betta1 = betta1;
        param.betta2 = betta2;
        param.sigma = sigma;
        param.gamma = gamma;
        bits = BITS(bi);
        %% solve objective function
        [U1, A1, U2, A2, H, W1, W2, Q] = solveDLSH(I_temp, T_temp, L_tr, bits, K1, K2, param);
        
        %% extend to the whole database
        test_codeX1 = (Phi_testI*U1') / (U1*U1');
        test_codeX2 = (Pht_testT*U2') / (U2*U2');
        test_hashX1 = test_codeX1*W1;
        test_hashX2 = test_codeX2*W2;
        %% calculate hash codes
        Yi_tr = sign((bsxfun(@minus, H , mean(H,1))));
        Yi_te = sign((bsxfun(@minus, test_hashX1 , mean(H,1))));
        Yt_tr = sign((bsxfun(@minus, H , mean(H,1))));
        Yt_te = sign((bsxfun(@minus, test_hashX2 , mean(H,1))));
        
        %% evaluate
        Dhamm_im = hammingDist(Yi_te+2, Yt_tr+2);
        Dhamm_te = hammingDist(Yt_te+2, Yi_tr+2);
        [mAPi2t] = perf_metric4Label( L_tr, L_te, Dhamm_im' );
        [mAPt2i] = perf_metric4Label( L_tr, L_te, Dhamm_te' );
        map(i,1) = mAPi2t;
        map(i,2) = mAPt2i;
    end
    fprintf('%d bits average map over %d runs for I->T: %.4f\n', bits, run, mean(map(:,1)));
    fprintf('%d bits average map over %d runs for T->I: %.4f\n\n\n', bits, run, mean(map(:,2)));
end
