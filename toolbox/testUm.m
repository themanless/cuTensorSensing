U = ones(3, 1, 2);
U(:,:, 1) = [1, 2, 3];
U(:,:, 2) = [1.1, 2.1, 3.1];
U_ = Tensor2fullCircM(U);
Um = mat2diaMat(U_, 2);
A = ones(2, 12);
for i=7:12
    A(:, i) = [2 2];
end
W = A*Um;
