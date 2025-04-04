import torch
from torchdeq import get_deq

import numpy as np
import matplotlib.pyplot as plt

deq = get_deq(f_solver='broyden', f_max_iter=200, f_tol=1e-9)

# The first equilibrium function
f = lambda z: torch.cos(z)
z0 = torch.tensor(0.0)
z_out, info = deq(f, z0)

f_abs_trace = info['abs_trace']
f_abs_trace = f_abs_trace.mean(dim=0)[1:]

# The second equilibrium function
g = lambda z: 0.5 * (z + 2 / z)
z0 = torch.tensor(0.5)
z_out, info = deq(g, z0)

g_abs_trace = info['abs_trace']
g_abs_trace = g_abs_trace.mean(dim=0)[1:]

# The third equilibrium function
h = lambda z: torch.exp(-z)
z0 = torch.tensor(0.5)
z_out, info = deq(h, z0)

h_abs_trace = info['abs_trace']
h_abs_trace = h_abs_trace.mean(dim=0)[1:]

# Convergence Visualization
iterations = np.arange(len(f_abs_trace))+1

plt.plot(iterations, f_abs_trace, 'o-', color='#4f96b8', markersize=4, linewidth=2, label='$f(z) = \cos(z)$')
plt.plot(iterations, g_abs_trace, 'o-', color='#ff9d00', markersize=4, linewidth=2, label='$g(z) = 0.5(z + 2/z)$')
plt.plot(iterations, h_abs_trace, 'o-', color='#3f8f31', markersize=4, linewidth=2, label='$h(z) = e^{-z}$')

plt.grid(True, which="both", ls="--", c='0.7')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('$\|z - f(z)\|$', fontsize=12)
plt.yscale('log')
plt.xticks(iterations)

plt.title('Fixed Point Convergence', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()

plt.savefig("test.png")
