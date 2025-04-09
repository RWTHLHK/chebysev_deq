import torch
import matplotlib.pyplot as plt
class WendlandQuintic(torch.nn.Module):
    def __init__(self, h=1.0):
        super().__init__()
        self.h = h
        self.radius_scale = 2.0  # max support radius
        self.fac = 7.0 / (4.0 * torch.pi * h**2)  # 2D normalization

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        q = r / self.h
        tmp = 1.0 - 0.5 * q
        val = torch.zeros_like(q)
        val[q < 2.0] = self.fac * (tmp[q < 2.0 ] ** 4) * (2 * q[q < 2.0] + 1)
        return val

    def dwdq(self, r: torch.Tensor) -> torch.Tensor:
        """Derivative dW/dq, fully masked like PySPH"""
        q = r / self.h
        tmp = 1.0 - 0.5 * q

        # initialize
        val = torch.zeros_like(r)

        # compactly support
        mask = (r < 2 * self.h) & (r > 1e-12)
        val[mask] = -5.0 * q[mask] * (tmp[mask] ** 3)

        return self.fac * val

    def d2wdq2(self, r: torch.Tensor) -> torch.Tensor:
      q = r / self.h
      tmp = 1.0 - 0.5 * q

      val = torch.zeros_like(r)
      mask = (r < 2 * self.h) & (r > 1e-12)

      val[mask] = -5.0 * (tmp[mask] ** 2) * (1.0 - 2.0 * q[mask])

      return self.fac * val


    def grad(self, xij: torch.Tensor) -> torch.Tensor:
        """
        Spatial gradient ∇W, based on xij
        xij: [E, 2] relative position vectors
        returns: [E, 2] gradient vectors
        """
        r = torch.norm(xij, dim=1)  # [E]
        dwdq = self.dwdq(r)         # already masked internally

        coef = torch.zeros_like(r)
        coef[r>1e-12] = dwdq[r>1e-12] / (self.h * r[r>1e-12])
        grad = coef.unsqueeze(1) * xij     # [E, 2]
        return grad

    def laplacian(self, xij: torch.Tensor) -> torch.Tensor:
      r = torch.norm(xij, dim=1)
      val = torch.zeros_like(r)

      q = r / self.h

      dwdq = self.dwdq(r)
      d2wdq2 = self.d2wdq2(r)
      val = torch.zeros_like(r)
      val[r>1e-12] = (d2wdq2[r>1e-12] + dwdq[r>1e-12] / q) / (self.h ** 2)
      return val


kernel = WendlandQuintic()

# 创建测试向量对：E个二维向量
xij = torch.tensor([
    [0.5, 0.0],
    [1.0, 0.0],
    [1.5, 0.0],
    [1.0, 1.0],
    [0.0, 0.0],  # 零向量情况
], dtype=torch.float32, requires_grad=True)

# Analytical gradient
grad_analytical = kernel.grad(xij)

# Numerical gradient (central difference)
epsilon = 1e-4
grad_numerical = torch.zeros_like(xij)

for i in range(xij.shape[0]):
    for j in range(2):
        xij_eps_plus = xij.clone().detach()
        xij_eps_plus[i, j] += epsilon
        xij_eps_minus = xij.clone().detach()
        xij_eps_minus[i, j] -= epsilon

        r_plus = torch.norm(xij_eps_plus[i])
        r_minus = torch.norm(xij_eps_minus[i])

        W_plus = kernel(r_plus.unsqueeze(0))
        W_minus = kernel(r_minus.unsqueeze(0))

        grad_numerical[i, j] = (W_plus - W_minus) / (2 * epsilon)

#compare
print("Analytical grad:\n", grad_analytical)
print("Numerical grad:\n", grad_numerical)
print("Difference:\n", grad_analytical - grad_numerical)

# 采样点
r = torch.linspace(0.01, 1.99, 500)  # 避免0，避免超出支撑范围
xij = torch.stack([r, torch.zeros_like(r)], dim=1)  # x方向相对位移向量

# 解析 laplacian
lap_analytic = kernel.laplacian(xij)

# 数值 laplacian：中心差分模拟 ΔW(r) = d²W/dr² + 1/r * dW/dr
eps = 1e-4
r_plus = r + eps
r_minus = r - eps

W = kernel(r)
W_plus = kernel(r_plus)
W_minus = kernel(r_minus)

dW_dr = (W_plus - W_minus) / (2 * eps)
d2W_dr2 = (W_plus - 2 * W + W_minus) / (eps ** 2)

lap_numeric = d2W_dr2 + (1 / r) * dW_dr

# error
error = lap_numeric - lap_analytic.detach()

# visulize
plt.figure()
plt.plot(r, lap_analytic.detach(), label='Analytical Laplacian')
plt.plot(r, lap_numeric, '--', label='Numerical Laplacian')
plt.legend()
plt.title('Laplacian Comparison')
plt.xlabel('r')
plt.ylabel('∆W')
plt.grid(True)
plt.savefig("laplacian.png")

plt.figure()
plt.plot(r, error, label='Error')
plt.title('Error (Numerical - Analytical)')
plt.xlabel('r')
plt.ylabel('Error')
plt.grid(True)
plt.savefig("laplacian_error.png")
