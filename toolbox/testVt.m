m = 3;
n = 2;
r = 1;
k = 2;
d = 4;
A = ones(4, 12);
A(:, 7:12) = 2*ones(4, 6);
X = ones(3, 2, 2);
y = 18*ones(d, 1);
U = ones(6, 2);
U_ = mat2diaMat(U,n);         % 编排后的矩阵 U    
W = A*U_;
V_ = W \ y ;

V = Vec2Mat(V_,[r*k, n]);       % 将得到的向量  V  重新转化为矩阵
