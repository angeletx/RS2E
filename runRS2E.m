
%% Load data
dataFile = './data/Smote.mat';
load(dataFile);

%% Set parameter
classLabels = unique(gnd);
classNumber  = numel(classLabels);

numberOfTrainPerClass = 5;
intraClassK = numberOfTrainPerClass-1;
interClassK = numberOfTrainPerClass-1;
dim = 4;
gamma = 1000;
mu = 0.01;
alpha = 0.001;
beta = 0.001;
wIter = 100;

f = numberOfTrainPerClass;
rng('default');
trainX = [];
trainY = [];
testX  = [];
testY  = [];
for j = 1 : classNumber
    indices = find(gnd==classLabels(j));
    randIndices = randperm(numel(indices));
    trainX = [trainX; fea(indices(randIndices(1 : f)),:)];
    trainY = [trainY; gnd(indices(randIndices(1 : f)))];
    testX  = [testX ; fea(indices(randIndices(f +1 : end)),:)];
    testY  = [testY ; gnd(indices(randIndices(f +1 : end)))];
end

CombinSym = RS2E(trainX', trainY, intraClassK, interClassK, dim, gamma);

[W, ~, ~, Zs, ~, ~] = optimize(trainX', dim, alpha, beta, mu, CombinSym, wIter);

trainZ = Zs;
testZ = W'* testX';

%% nearest knn classifier
knnClassifier = fitcknn(trainZ', trainY, 'NumNeighbors', 1, 'Distance', 'euclidean');
testPred = predict(knnClassifier, testZ');

% Compute accuracy
testAccuracy = sum(testPred == testY) / numel(testY);

fprintf('Test Accuracy: %f\n', testAccuracy);

%%
function CombinSym = RS2E(trainX, trainY, intraClassK, interClassK, dim, gamma, dopt)
[~, N] = size(trainX);

if ~exist('dopt', 'var')
    dopt = 'euclidean';
end

if strcmp(dopt, 'seuclidean')
    disp('seuclidean')
    distance=pdist2(trainX',trainX','seuclidean');
else
    distance=pdist2(trainX',trainX','euclidean');
end

[~,index] = sort(distance);

nbLabel = trainY(index);
nbWithin = zeros(intraClassK, N);

for i= 1 : N
    oneIndexWithin = index(:, i);
    oneIndexWithin(nbLabel(:,i) ~= trainY(i)) = [];

    % center is included
    nbWithin(:,i) = oneIndexWithin(1:intraClassK);

end

Mw = calcu_opt_matrix(trainX, nbWithin, dim);

Mw_new= trainX*Mw*trainX';

%% Direct minus strategy
E = calcu_E_Bpist_Global(trainX,trainY,interClassK,distance);

%% Indirect Laplacian strategy

Combin= Mw_new / (intraClassK * intraClassK * N * dim) - gamma*E/ (interClassK * N * dim);

CombinSym = (Combin+Combin')/2;

end


%%%%%%%%

function [W, obj, deltaW, Z, R,criterion]= optimize(X, dim, alpha, beta, mu, Combin, wIter)

maxIter = 100;

[d,n] = size(X);
W = zeros(d,dim);
Z = zeros(dim,n);
R = zeros(dim,n);
Y1 = zeros(dim,n);

% mu= 1e-2;
% max_mu =1e3;
% rho=1.1;

deltaW = zeros(maxIter,1);
deltaR = zeros(maxIter,1);
deltaZ = zeros(maxIter,1);
criterion = zeros(maxIter,1);
obj = zeros(maxIter,1);

% If mu is fixed
Psi = Combin + mu*X*X';

%% ======================two methods: low-rank and sparse====================
for iter = 1: maxIter

    if mu > eps
        invMu = 1/mu;
    else
        invMu = 0;
    end

    % ===============================Update W==================================
    preW = W;

    Phi = X*(Z' + R' - Y1'*invMu);
    [W, objW, delta] = solveW(Psi, Phi, mu, wIter);

    deltaW(iter) = norm(W-preW, 'fro')/norm(preW, 'fro');

    % ===============================Update Z==================================
    preZ = Z;

    Z = solveZ(W'*X - R + Y1*invMu, beta*invMu);

    deltaZ(iter) = norm(Z-preZ, 'fro')/norm(Z, 'fro'); %norm(Z-preZ, 'fro');

    % ===============================Update R==================================

    preR = R;

    Q= W'*X - Z + Y1*invMu;

    R = solveR(Q, alpha*invMu);

    deltaR(iter) = norm(R-preR, 'fro')/norm(preR, 'fro'); 

    %% Compute equality
    leq1 = W'*X - Z - R;

    %% covergence of objective
    obj(iter) = 0.5*trace(W'*Combin*W) + alpha*sum(sqrt(sum(R.^2, 1))) + beta*sum(svd(Z)); % + mu/2*(norm(leq1+Y1*invMu, 'fro'));

    % ===============================Update Parameter===================================
    Y1 = Y1 + mu*leq1;

    % ===============================Convergence condition=================================
    %     stopALM = norm(leq1,'fro');
    %     criterion(iter) = stopALM;
    %     if stopALM < 1e-6
    %         % disp([num2str(iter), ': break']);
    %         break
    %     end
end

end

%%
% solve min 0.5*||Z-J||_F^2 + gamma*||Z||_*
%
function Z = solveZ(J, gamma)

[U,sigma,V] = svd(J,'econ');
sigma = max(0, diag(sigma)-gamma);
Z = U * diag(sigma) * V';

end

%%
% solve min 1/2*tr(X^TAX) - alpha*tr(X^TB)
%
function [X, obj, delta] = solveW(A, B, alpha, maxIteration)

if ~exist('maxIteration', 'var')
    maxIteration = 500;
end
minDelta = 1E-8;

if size(B, 1) < size(B, 2)
    error('size(B, 1) < size(B, 2)');
end

A = (A + A') / 2;

gamma = abs(eigs(A,1));

dim = size(B, 2);
X = orth(rand(size(B)));
Atilde = gamma *eye(size(A)) - A;
alphaB = alpha * B;
obj = zeros(1, maxIteration);
delta = zeros(1, maxIteration);

if norm(alphaB, 'fro') < eps
    [U, S] = eig(A);
    [~, eindex] = sort(diag(S));
    X = U(:, eindex(1:dim));
    return;
end

for t = 1 : maxIteration

    M = Atilde * X + alphaB;

    [U, ~, V] = svd(M,'econ');
    X = U * V';
    obj(t) = trace(0.5 * X' * A * X - X' * alphaB);

    if t > 1
        delta(t) = abs(obj(t) - obj(t-1));
        if (delta(t) <= minDelta)
            obj = obj(1:t);
            delta = delta(1:t);
            return;
        end
    end

end

end

%% solve gamma*||R||_{2,1} + 0.5||R-Q||_F^2

function R = solveR(Q, gamma)

normQCol = sqrt(sum(Q.^2, 1));

positiveIndex = (normQCol > gamma);
normQCol(positiveIndex) = (normQCol(positiveIndex) - gamma) ./ normQCol(positiveIndex);
normQCol(~positiveIndex) = 0;
diag_Q = diag(normQCol);

R=Q*diag_Q;

end

%%%%%%%%%%%%

function M = calcu_opt_matrix(X, nb, dst_dim)
% X:         the source point in sample space. Each column is a data point
% nb:        the neighboring relationship
% dst_dinm:  the lower dimensionality to be reduced

%return:     A sparse matrix

[K, N] = size(nb);

nMax = K * K * N;
M   =   spalloc(N, N, nMax);

for i = 1 : N
    ind = nb(:, i);

    % now perfrom the local PCA and obtaine the local error matrix:
    thisx = X(:, ind);                        % get the local patch
    V = local_pca(thisx, dst_dim);                 % claculate the local coordinates

    ipl = calcu_spline_matrix(V);

    M(ind, ind) = M(ind, ind) + ipl;
end
end

%%
function V = local_pca(X, d)
% X=thisx
% X: each colmun is a data point
% d: the dimensioanlity of low-dimensioanl space

% return
%V:   the local coordinate after performing local PCA, each coulmn is a data point

[D, K] = size(X);

X = X + 0.0001 * randn(D, K);         % you can consider to add some noise, if some neighbors are exactly equal to each other

thisx = X - repmat(mean(X')',1, K) ;   %  centerized

% fast
if D <= K
    [U, DD, Vpr] = svd(thisx);            % Vpr: each column is a eigenvector

    V = U(:, 1:d)' * thisx; %*10;                % each row is a selected eigenvector, here scale it for computation
else
    CC = thisx * thisx';
    options.disp = 0; options.isreal = 1; options.issym = 1;
    [W, evalW] = eig(CC);
    [~, eindex] = sort(diag(evalW));
    U = W(:, eindex(1:d));
    V = U' * thisx;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E =calcu_E_Bpist_Global(X, label, interClassK, distance)

[D, N] = size(X);

[~,index] = sort(distance);

nbLabel = label(index);

nbBetween = zeros(interClassK, N);
for i= 1 : N
    oneIndexBetween = index(:, i);

    oneIndexBetween(nbLabel(:,i) == label(i)) = [];
    nbBetween(:,i) = oneIndexBetween(1:interClassK);
end

[K, N] = size(nbBetween);

ind = nbBetween;
Xnew_b = X(:, ind);
Xnew_p = kron(X,ones(1,K));
E=Xnew_p*Xnew_p'-2*Xnew_b*Xnew_p'+Xnew_b*Xnew_b';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LL = calcu_spline_matrix(X)
%    X: all the points in a nieghborhood, here thay are expressed in low-dimensional space
%    In X, each column is a data point,

%    LL:    return the inverse matrix


[d, num] = size(X);

dd = num + 1 + d;             % for each components there have dd parameters to be estimated

W = zeros(d, dd);               % the final parameter matrix

X2 = sum(X.^2, 1);
distance = repmat(X2, num, 1) + repmat(X2', 1, num) - 2 * X'* X;

% adopt the spline
distance   = distance .* log(distance + 0.00001) + 0.0001 * eye(num);

distance = 0.5 * distance;      %acutally, you can just delete this sentence


%construct the coefficient matrix
P = ones(num, 1);
P = [P, X'];                        % add the column, including the source data
C = zeros(d + 1, d + 1);

L = [distance, P; P', C];          % construct the coefficient matrix

LL = pinv(L);

LL = LL(1:num, 1: num);

end


