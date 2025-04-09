import numpy as np
import matplotlib.pyplot as plt
import ptwt
import pywt

def cheb(N):
    """
    构造 Chebyshev 微分矩阵 D 和 Chebyshev 节点 x
    参考 Trefethen 的谱方法代码
    """
    if N == 0:
        D = np.array([[0]])
        x = np.array([1.0])
        return D, x
    x = np.cos(np.pi * np.arange(N+1) / N)  # Chebyshev 节点（在[-1,1]区间）
    c = np.hstack([2, np.ones(N-1), 2]) * (-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1))
    dX = X.T - X
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

# 设置节点数量
# N = 16
# D, x = cheb(N)

# # 定义函数及其解析导数
# f = np.exp(x)
# deriv_exact = np.exp(x)

# # 利用 collocation 方法：直接用微分矩阵作用于函数在节点处的值
# deriv_collocation = D @ f

# # 输出结果
# print("Chebyshev 节点 x:")
# print(x)
# print("\n函数值 f = sin(pi*x):")
# print(f)
# print("\nCollocation 方法计算的导数 D@f:")
# print(deriv_collocation)
# print("\n解析导数 pi*cos(pi*x):")
# print(deriv_exact)
# print("\n误差 (computed - exact):")
# print(deriv_collocation - deriv_exact)

# # 绘图比较
# plt.figure(figsize=(8,4))
# plt.plot(x, deriv_collocation, 'bo-', label='Collocation derivative')
# plt.plot(x, deriv_exact, 'rx--', label='Exact derivative')
# plt.xlabel('x')
# plt.ylabel("f'(x)")
# plt.legend()
# plt.title("Chebyshev Collocation Method for f'(x) with f(x)=sin(pi*x)")
# plt.savefig("/home/doelz-admin/projects/spectra_deq/coompare.png")

w = pywt.Wavelet('db3')
dec_lo, dec_hi, rec_lo, rec_hi = w.filter_bank
h0,h1,h2,h3,h4,h5 = dec_lo
A0 = np.array([[h5,0, 0, 0, 0],
               [h3,h4,h5,0, 0],
               [h1,h2,h3,h4,h5],
               [0, h0, h1,h2,h3],
               [0,0,0,h0,h1]])

phi, psi, x = w.wavefun(1)
print(phi)
phi0 = phi[0:10:2]
print(phi0)
phi0_rec = np.dot(A0, phi0)
print(phi0_rec)
# phi_rec = np.dot()
# print("scaling func: ", phi)
# print("wave fun: ", psi)
# print("sampling grid: ", x)
