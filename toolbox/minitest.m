m = 2;
r = 1;
n = 2;
k = 2;
d = 8;
IterNum = 5;
X = [1 1.1 2 2.1 3 3.1 4 4.1];
X_s = reshape(X, [4,2]);
A = zeros(8, 8);
for i=1:8
    A(i, :) = [1 0 0 1 0 0 0 1];
end
At = zeros(8, 8);
for i=1:8
    At(i, :) = [1 0 0 0 0 1 0 1];
end
y = A*X';
X_I = zeros(2, 2, 2);
error = zeros(IterNum,1);      %误差数组


%% 初始化 A2m U0

U0 = ones(m, r, k);
U = Tensor2fullCircM(U0)     % U 初始化时化为循环矩阵
% parpool open;
%% 算法迭代主体
for l = 1 : 1
    %V = LS_V(A_all, U, r*k, y, n);
    U_ = mat2diaMat(U,n)         % 编排后的矩阵 U    
    W = A*U_
    V_ = W \ y 
%         V_ = lsqr(W, y, 1e-6, 100); 
    clear W;
    V = Vec2Mat(V_,[r*k, n])       % 将得到的向量  V  重新转化为矩阵
    %U = LS_U(A_all_t, V, r*k, y, m*k);
    Vt = V.';
    Vt_ = mat2diaMat(Vt,m*k);
    W = A*Vt_
    U_ = W \ y 
%       U_ = lsqr(W, y, 1e-6, 100); 
    clear W;
    U = (Vec2Mat(U_,[r*k, m*k])).'
    X_ = U * V;
end

%% LS函数
