function [RSE,error]=TS_fft(X, A_all, y, IterNum, r, SamplingRate)
%% Simplified Alternating Minimization for tensor sensing
% Problem state：min|| b - <A,(U*V)> ||_F^2, s.t.:tubal-rank of U * V=r
% Input : X = U * V  - 理想数据，U:m * r * k,   V:r * n * k，    X：m * n * k
%         A - 采样tensor，每一个前切面A_i大小为mk * n,每一个A_i与X做内积，所得的数作为y_i,1 <= i<= d
%         IterNum - 算法迭代次数
%         r - target tubal-rank
% Output : U^c : mk * rk  -  X^s的左近似分解项
%          V^s : rk * n   -  X^s的右近似分解项
%          X^s : mk * n      X^s = U^c * V^s

error = zeros(IterNum,1);      %误差数组


%% 初始化 A2m U0

[m, n, k] = size(X);
U = randn(m, r, k);
Xre = A_all \ y;
X_re = vec2Tensor(Xre, m, n, k);
X_s = X_re;
U_f = fft(U, [], 3);
V_f = zeros(r, n, k);
%U = Tensor2fullCircM(U0);     % U 初始化时化为循环矩阵
% parpool open;
%% 算法迭代主体
for l = 1 : IterNum
    %T = 1/2((A\y) + tprod(U, V));
    X_s = (X_re + X_s)./2;
    X_f = fft(X_s, [], 3);
    for i=1:k
        V_f(:,:, i) = U_f(:, :, i) \ X_f(:, :, i);

        U_f(:, :, i) = (V_f(:,:,i)' \ (X_f(:, :, i)'))';
        X_f(:, :, i) = U_f(:, :, i) * V_f(:, :, i);
    end
    X_s = ifft(X_f, [], 3);
    %V = LS_V(A_all, U, r*k, y, n);
    %U = LS_U(A_all_t, V, r*k, y, m*k);
    %X_ = U * V;
    RSE = norm(X_s(:) - X_re(:)) / norm(X_re(:));
    fprintf('SamplingRate:%d,%d-th iteration,error is %e\n',SamplingRate,l,RSE);
    error(l,:) = RSE;
end

    function [T] = vec2Tensor(Tre, m, n, k)
        Tre = full(Tre);
        for i = 1:n
            for j = 1:m
                for z = 1:k
                    T(j, i, z) = Tre(((i-1)*m+j-1)*k+z);
                end
            end
        end
    end

end

