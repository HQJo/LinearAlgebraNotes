#! https://zhuanlan.zhihu.com/p/336931022
本文使用 jupyter notebook 和 Julia 写成，原始 notebook 可在 [github 仓库](https://github.com/HQJo/LinearAlgebraNotes) 找到。

## 特征值分解

上节讲到了矩阵的对角化，对角化的结果也叫做**特征值分解**。
$$
    A = S \Lambda S^{-1}
$$
其中 $\Lambda$ 对角元素即特征值，$S$ 的列向量 $\{u_i\}_{i=1}^{n}$即对应的特征向量，且这些特征向量构成一组基。

既然是基，那么任意向量 $x$ 都可以分解成它们的线性组合：$x = \sum_{i} \alpha_i u_i$，那么矩阵向量乘就可以变成：
$$
    A x = A \sum_{i} \alpha_i u_i = \sum_{i} \alpha_i A u_i = \sum_{i} \alpha_i \lambda_i u_i
$$
上式告诉我们一个有意思的结论：当我们将 $x$ 分解后，矩阵向量乘不仅限于其原始定义，还可以这样看待：**$x$ 中每有一单位的 $u_i$，经过 $A$ 变换后就变成了 $\lambda_i$ 倍的 $u_i$**。

这可以比喻成某种“查表”操作：先查询各特征向量上的分量，然后把对应分量系数 $\alpha_i$ 乘上 $\lambda_i$ 即可，每个特征向量上的分量都变为了原来的 $\lambda_i$ 倍。

举个例子辅助理解：假设基是颜色的三原色红绿蓝，$A$ 是一个对颜色的变换，它将红色减小为原来 0.8 倍，绿色保持不变，蓝色增强 1.2 倍。那么对任何一个颜色应用 $A$ 变换，相当于先将其分解成三原色，再进行如上操作，就能得到变换后的颜色。

然而仔细想想，这不过是把原来的困难放到了对 $x$ 分解这一步中，将 $x$ 表示成特征向量的线性组合可不是件容易的事。幸运的是，如果 $A$ 是一个对称阵，那么这件事就变得简单了。

如果 $A$ 是一个对称矩阵，由谱定理，它可以被正交对角化，也即 $S$ 是正交阵（$S S^{\mathrm{T}} = S^{\mathrm{T}} S = I$），此时特征向量不仅是一组基，还是一组**正交基**！正交基最大的好处在于，分量系数可以简单的通过内积得到：$\alpha_i = \langle x, u_i \rangle$。那么，如果我们事先知道了 $A$ 的特征向量，矩阵向量乘就能变得非常简单：

1. 得到各分量系数：$\alpha_i = \langle x, u_i \rangle$；
2. 各分量乘以对应特征值：$\beta_i = \alpha_i \cdot \lambda_i$；
3. 再进行线性组合：$A x = \sum_{i} \beta_i u_i$；

我们通过实际例子感受一下以上过程：


```julia
n = 4
A = mod.(rand(Int, n, n), 10)
A .+= A'
x = mod.(rand(Int, n), 10)
@prettyshow A
@prettyshow x
```

    A is 4×4 Array{Int64,2}:
      4  16   7   9
     16   6  14   3
      7  14   8  13
      9   3  13   4
    x is 4-element Array{Int64,1}:
     7
     5
     3
     0



```julia
# 特征值分解获得特征值与特征向量
λ, U = eigen(A)
@prettyshow U
```

    U is 4×4 Array{Float64,2}:
      0.523506   0.584648  -0.38216   -0.487936
     -0.598768  -0.26142   -0.530224  -0.54037
      0.42988   -0.60196    0.378433  -0.55645
     -0.427344   0.47696    0.655442  -0.400353


用两种方式计算矩阵向量乘 $A x$：


```julia
# 获得各分量上的系数
α = U' * x

# 矩阵向量乘结果
y1 = A * x
# 分量乘对应特征值，再线性组合
y2 = U * (α .* λ)
@prettyshow y1
@prettyshow y2
println(all(abs.(y1 .- y2) .< 1e-6))
```

    y1 is 4-element Array{Int64,1}:
     129
     184
     143
     117
    y2 is 4-element Array{Float64,1}:
     128.99999999999946
     183.99999999999932
     142.9999999999994
     116.99999999999993
    true


可以看到，两种方法计算的结果是一样的（除去计算误差）。

## 奇异值分解

上面我们都在讨论方阵，方阵才有特征值分解。如果是一般的矩阵 $A_{m\times n}$ 呢？我们有奇异值分解（Singular Value Decomposition, SVD)。

用公式来讲，矩阵 $A_{m\times n}$ 的奇异值分解是：
$$
    A = U \Sigma V^{\mathrm{T}}
$$
其中 $U$ 是 $m\times m$ 的正交阵，$\Sigma$ 是 $m\times n$ 的（伪）对角阵，$V^{\mathrm{T}}$ 是 $n\times n$ 的正交阵。

SVD 的内容十分丰富，这里只介绍 SVD 对于矩阵向量乘能带来什么新的认识。

SVD 的公式可以写作如下形式（这里假设 $m < n$，中间的 $\Sigma$ 没写全，没写出的部分自动补 0）：
$$
    A = [u_1, \dotsc, u_m]
    \begin{bmatrix} \sigma_1 & & \\ & \ddots & \\ & & \sigma_m \end{bmatrix}
    \begin{bmatrix} v_1^{\mathrm{T}} \\ \vdots \\ v_n^{\mathrm{T}} \end{bmatrix} = \sum_{i=1}^{m} \sigma_i u_i v_i^{\mathrm{T}}
$$

对于一个 $n$ 维向量 $x \in \mathbb{R}^{n}$，我们可以将其分解为 $\{v_i\}_{i=1}^{n}$ 的线性组合：$x = \sum_{j=1}^{n} \alpha_j v_j$。对于矩阵向量乘，我们有：
$$
\begin{equation}
    \begin{aligned}
        A x &= \sum_{i=1}^{m} \sigma_i u_i v_i^{\mathrm{T}} \sum_{j} \alpha_j v_j \\
        &= \sum_{i=1}^{m} \sum_{j=1}^{n}\sigma_i \alpha_j u_i v_i^{\mathrm{T}} v_j \\
        &= \sum_{i=1}^{m} \sigma_i \alpha_i u_i
    \end{aligned}
\end{equation}
$$
其中最后一步用到了 $\{v_i\}_{i=1}^{n}$ 是一组标准正交基。

以上推导能告诉我们什么呢？不同于方阵，$A$ 的作用是将一个 $\mathbb{R}^{n}$ 维向量变成一个 $\mathbb{R}^{m}$ 维向量。记 $y = Ax$，$y$ 一定能表示成 $\{u_i\}_{i=1}^{n}$ 的线性组合，但具体是多少呢？

SVD 告诉我们，**$x$ 中每有一个单位的 $v_i$，经过 $A$ 变换后，$y$ 中就有 $\sigma$ 倍的 $u_i$**。比喻成查表操作的话，我们先查询 $x$ 在 $\{v_i\}_{i=1}^{n}$ 下的分量，把分量乘上 $\sigma_i$，把 $v_i$ 换成 $u_i$，就得到了 $Ax$ 的结果。

可以将这个结论与特征值分解的作比较，特征值分解由于是方阵，是同一个空间中的变换，所以只涉及同一组基；而非方阵涉及一个空间到另一个空间，所以是两组基之间的关系。

老样子，还是通过实际例子验证一下：


```julia
m, n = 3, 4
A = mod.(rand(Int, m, n), 10)
x = mod.(rand(Int, n), 10)
@prettyshow A
@prettyshow x
```

    A is 3×4 Array{Int64,2}:
     8  5  4  5
     7  8  8  3
     9  8  0  1
    x is 4-element Array{Int64,1}:
     0
     6
     8
     1



```julia
U, Σ, Vt = svd(A)
@prettyshow U
@prettyshow Σ
@prettyshow Vt
```

    U is 3×3 Array{Float64,2}:
     -0.541792   0.121956  -0.831618
     -0.641885   0.578725   0.503052
     -0.542628  -0.806352   0.235267
    Σ is 3-element Array{Float64,1}:
     20.410756119081146
      5.916286839632225
      3.224683655583361
    Vt is 4×3 Adjoint{Float64,Array{Float64,2}}:
     -0.671763  -0.377001  -0.314504
     -0.596992  -0.204729   0.542212
     -0.357765   0.865006   0.216439
     -0.253653   0.260231  -0.748497



```julia
# 得到各 v_i 分量下的系数
α = Vt' * x

# 矩阵向量乘结果
y1 = A * x
# 分量乘对应特征值，再对 u_i 线性组合
y2 = U * (α .* Σ)

@prettyshow y1
@prettyshow y2
println(all(abs.(y1 .- y2) .< 1e-6))
```

    y1 is 3-element Array{Int64,1}:
      67
     115
      49
    y2 is 3-element Array{Float64,1}:
      67.00000000000001
     115.00000000000003
      49.00000000000003
    true


可以看到，两种方法计算的结果是一样的（除去计算误差）。
