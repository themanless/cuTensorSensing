function [RSE,error]=TS_fft(X, A_all, y, IterNum, r, SamplingRate)
%% Simplified Alternating Minimization for tensor sensing
% Problem state��min|| b - <A,(U*V)> ||_F^2, s.t.:tubal-rank of U * V=r
% Input : X = U * V  - �������ݣ�U:m * r * k,   V:r * n * k��    X��m * n * k
%         A - ����tensor��ÿһ��ǰ����A_i��СΪmk * n,ÿһ��A_i��X���ڻ������õ�����Ϊy_i,1 <= i<= d
%         IterNum - �㷨��������
%         r - target tubal-rank
% Output : U^c : mk * rk  -  X^s������Ʒֽ���
%          V^s : rk * n   -  X^s���ҽ��Ʒֽ���
%          X^s : mk * n      X^s = U^c * V^s

error = zeros(IterNum,1);      %�������


%% ��ʼ�� A2m U0

[m, n, k] = size(X);
U = randn(m, r, k);
Xre = A_all \ y;
X_re = vec2Tensor(Xre, m, n, k);
X_s = X_re;
U_f = fft(U, [], 3);
V_f = zeros(r, n, k);
%U = Tensor2fullCircM(U0);     % U ��ʼ��ʱ��Ϊѭ������
% parpool open;
%% �㷨��������
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

