import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from torch_geometric.data import Data
import torch
from torch_geometric.transforms import RadiusGraph

# gen domain
domain_size = 1.0
hole_radius = 0.2
num_points = 20  # 网格分辨率

# 生成均匀网格点
x = np.linspace(0, domain_size, num_points)
y = np.linspace(0, domain_size, num_points)
xx, yy = np.meshgrid(x, y)
points = np.vstack((xx.ravel(), yy.ravel())).T

# 圆弧判断函数（右下1/4圆）
def is_in_quarter_circle(p, center=(1.0, 0.0), radius=0.2):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    return (dx**2 + dy**2 <= radius**2) and (dx <= 0) and (dy >= 0)

# 过滤掉圆弧区域内的点
filtered_points = np.array([p for p in points if not is_in_quarter_circle(p)])

# 沿四分之一圆弧采样额外点用于轮廓显示
theta = np.linspace(np.pi / 2, np.pi, 30)  # 90 到 180 度
arc_x = 1.0 + hole_radius * np.cos(theta)
arc_y = 0.0 + hole_radius * np.sin(theta)
arc_points = np.vstack((arc_x, arc_y)).T

# 合并用于轮廓显示的采样点
all_points = np.vstack((filtered_points, arc_points))

# 绘图
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(all_points[:, 0], all_points[:, 1], s=20, color='blue', label='Sampling Points + Arc')

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect('equal')
# ax.set_title("Sampling Points with Quarter-Circle Hole + Arc Outline")
# ax.legend()
# plt.grid(True)
# plt.savefig("domain.png")
x_vals = all_points[:, 0]
y_vals = all_points[:, 1]
def u_exact(x, y):
    return 1 + x**2 + 2 * y**2

u_gt = u_exact(x_vals, y_vals)

r = domain_size / num_points
r_ball = 3*r

# 假设 all_points 是 numpy 数组 [N, 2]
pos = torch.tensor(all_points, dtype=torch.float)
data = Data(pos=pos)
transform = RadiusGraph(r=r_ball, loop=False)

# 应用 transform，构造 edge_index
data = transform(data)
print(data)
