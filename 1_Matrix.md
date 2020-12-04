#! https://zhuanlan.zhihu.com/p/330557394
```julia
using LinearAlgebra;
using Random;
Random.seed!(42);
```


```julia
function prettyshow(x::AbstractArray)
    show(stdout, MIME"text/plain"(), x)
    println()
    nothing
end
```




    prettyshow (generic function with 1 method)



## 矩阵

什么是矩阵？最简单的理解，一个 $m\times n$ 的矩阵是把 $m \times n$ 个数排成一个 $m$ 行 $n$ 列的表。

例如，下面是一个 $3 \times 4$ 的矩阵 $A$。


```julia
m, n = 3, 4;
A = mod.(rand(Int, m, n), 10)
```




    3×4 Array{Int64,2}:
     4  3  7  8
     7  6  3  6
     6  2  1  5



第二个角度，我们可以从行/列向量的角度去理解。一个 $m\times n$ 的矩阵可以看成 $m$ 个行向量或 $n$ 个列向量的堆叠：
$$
\begin{equation}
\begin{aligned}
A &= \begin{bmatrix} \mathbf{b}_1^\mathrm{T} \\ \mathbf{b}_2^\mathrm{T} \\ \vdots \\ \mathbf{b}_m^\mathrm{T} \end{bmatrix} \\
&= [\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n]
\end{aligned}
\end{equation}
$$

$\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n$ 是 $n$ 个 $\mathbb{R}^{m}$ 的列向量，$\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_m$ 是 $m$ 个 $\mathbb{R}^{n}$ 的行向量。

在上面的例子中，每个行向量和列向量分别是：


```julia
for (i, b) in enumerate(eachrow(A))
    println("$(i)-th row vector: ", b)
end
println()
for (i, a) in enumerate(eachcol(A))
    println("$(i)-th column vector: ", a)
end
```

    1-th row vector: [4, 3, 7, 8]
    2-th row vector: [7, 6, 3, 6]
    3-th row vector: [6, 2, 1, 5]

    1-th column vector: [4, 7, 6]
    2-th column vector: [3, 6, 2]
    3-th column vector: [7, 3, 1]
    4-th column vector: [8, 6, 5]


第三个角度，我们可以从线性变换的角度去理解。$A$ 矩阵代表了一种从 $\mathbb{R}^{n}$ 空间到 $\mathbb{R}^{m}$ 空间的变换，具体的变换通过乘矩阵 $A$ 实现。任何一个 $\mathbb{R}^{n}$ 中的向量与 $A$ 相乘，就变换成了一个 $\mathbb{R}^{m}$ 中的向量。


```julia
x =  mod.(rand(Int, n), 10)
println(length(x), "\t", x)
y = A * x
println(length(y), "\t", y)
```

    4	[6, 8, 0, 8]
    3	[112, 138, 92]



```julia

```

## 矩阵向量乘

矩阵向量乘也可以从对应的三种角度去看。

第一种，从单纯的数表角度，矩阵向量乘就是一系列乘法和加法。

$$
A\mathbf{x} = \mathbf{y} \qquad \mathbf{y}(i) = \sum_{j=1}^{n} A(i, j) \mathbf{x}(j) \quad \forall\ i = 1,2,\ldots, m
$$


```julia
prettyshow(A)
prettyshow(x)
prettyshow(y)
println()

# 验证 y 的每一个元素
for i in 1:m
    print("$(y[i]) = ")
    for j in 1:n
        print("$(A[i, j])×$(x[j])", j == n ? "" : " + ")
    end
    println()
end
```

    3×4 Array{Int64,2}:
     4  3  7  8
     7  6  3  6
     6  2  1  5
    4-element Array{Int64,1}:
     6
     8
     0
     8
    3-element Array{Int64,1}:
     112
     138
      92

    112 = 4×6 + 3×8 + 7×0 + 8×8
    138 = 7×6 + 6×8 + 3×0 + 6×8
    92 = 6×6 + 2×8 + 1×0 + 5×8


第二种角度，从列向量的线性组合来看，矩阵乘向量就是用向量的元素对列向量作线性组合。

假设 $A\cdot \mathbf{x}$， $A$ 是 $m\times n$ 的矩阵 ，有 $n$ 个列向量；$\mathbf{x}$ 是 $\mathbb{R}^{n}$ 的向量，刚好有 $n$ 个值。用这些值乘以对应列向量再相加，即得到运算的结果 $A\cdot \mathbf{x} = \mathbf{y}$。

$$
    A\cdot \mathbf{x} = [\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n] \cdot [x_1, x_2, \ldots, x_n]^{\mathrm{T}} = \mathbf{x}(1) \cdot a_1 + \mathbf{x}(2) \cdot a_2 + \cdots \mathbf{x}(n) \cdot a_n
$$


```julia
# 矩阵向量乘结果：
prettyshow(y)
println()

# 线性组合结果：
y2 = zero(y)
for i in 1:n
    y2 += x[i] * A[:, i]
end
prettyshow(y2)

print(y2, " = ")
for i in 1:n
    print("$(x[i])×$(A[:, i])", i == n ? "" : " + ")
end
```

    3-element Array{Int64,1}:
     112
     138
      92

    3-element Array{Int64,1}:
     112
     138
      92
    [112, 138, 92] = 6×[4, 7, 6] + 8×[3, 6, 2] + 0×[7, 3, 1] + 8×[8, 6, 5]

第三种角度，则是线性变换，矩阵向量乘将 $\mathbb{R}^{n}$ 中的向量变换成了 $\mathbb{R}^{m}$ 中的向量。

另外，两个矩阵相乘，也可以从行/列向量角度去看。设 $A \in \mathbb{R}^{m\times n}$, $B \in \mathbb{R}^{n \times p}$：

$AB$ 相乘的结果，其列向量即 $A$ 左乘 $B$ 矩阵的每一个列向量：
$$
    AB = A [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_p] = [A \mathbf{b}_1, A \mathbf{b}_2, \ldots, A \mathbf{b}_p]
$$

$AB$ 相乘的结果，其行向量即 $B$ 右乘 $A$ 矩阵的每一个行向量：
$$
    AB = \begin{bmatrix} \mathbf{a}_1^{\mathrm{T}} \\ \mathbf{a}_2^{\mathrm{T}} \\ \vdots \\ \mathbf{a}_m^{\mathrm{T}} \end{bmatrix} B = \begin{bmatrix} \mathbf{a}_1^{\mathrm{T}} B \\ \mathbf{a}_2^{\mathrm{T}} B \\ \vdots \\ \mathbf{a}_m^{\mathrm{T}} B \end{bmatrix}
$$
