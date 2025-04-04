import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdeq import get_deq
import matplotlib.pyplot as plt

class FixedPointResNet(nn.Module):
    def __init__(self, u_in_channels=2, z_in_channels=5, hidden_channels=32, num_groups=8):
        """
        A fixed-point residual network for updating the displacement field u.
        It processes u (B, N, N, 2) and a combined z (B, N, N, 5) separately and fuses them.
        GroupNorm layers are added after each convolution.
        Output is a new u of shape (B, N, N, 2).
        """
        super().__init__()
        # Branch for u: project u (2 channels) to hidden features.
        self.conv_u = nn.Conv2d(u_in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm_u = nn.GroupNorm(num_groups, hidden_channels)
        # Branch for z: project z (5 channels) to hidden features.
        self.conv_z = nn.Conv2d(z_in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm_z = nn.GroupNorm(num_groups, hidden_channels)
        # Fusion layer to combine the two branches.
        self.conv_fuse = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm_fuse = nn.GroupNorm(num_groups, hidden_channels)
        # Output projection: bring hidden features back to 2 channels (for displacement update).
        self.conv_out = nn.Conv2d(hidden_channels, u_in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_u.weight.data.normal_(0, 0.01)
        self.conv_z.weight.data.normal_(0, 0.01)
        self.conv_fuse.weight.data.normal_(0, 0.01)
        self.conv_out.weight.data.normal_(0, 0.01)

    def forward(self, u, z):
        """
        Args:
            u: Displacement field, shape (B, N, N, 2).
            z: Combined tensor (residual R(u) and material parameters), shape (B, N, N, 5).
        Returns:
            new_u: Updated displacement field, shape (B, N, N, 2).
        """
        # Convert inputs from (B, N, N, C) to (B, C, N, N)
        B, Nx, Ny, _ = u.shape
        u_feat = u.permute(0, 3, 1, 2)  # (B, 2, N, N)
        z_feat = z.permute(0, 3, 1, 2)  # (B, 5, N, N)

        # Process u and z separately with GroupNorm.
        h_u = F.relu(self.norm_u(self.conv_u(u_feat)))  # (B, hidden_channels, N, N)
        h_z = F.relu(self.norm_z(self.conv_z(z_feat)))    # (B, hidden_channels, N, N)

        # Fuse the features.
        h = h_u + h_z
        h = F.relu(self.norm_fuse(self.conv_fuse(h)))

        # Compute an update for u.
        delta_u = self.conv_out(h)                      # (B, 2, N, N)
        # Fixed point iteration: new u = u + delta_u.
        # alpha = alpha_out.permute(0, 2, 3, 1).view(B, Nx, Ny, 2, 2)
        # R_extracted = z_feat[:, :2, :, :].permute(0, 2, 3, 1)  # (B, N, N, 2)
        # delta_u = torch.matmul(alpha, R_extracted.unsqueeze(-1)).squeeze(-1)  # (B, N, N, 2)
        # # Convert back to (B, N, N, 2)
        new_u = u_feat + delta_u
        new_u = new_u.permute(0, 2, 3, 1)
        return new_u

def build_z(R, material_params):
    """
    Concatenate residual R (shape: (B, N, N, 2)) with material parameters (shape: (B, 3))
    after expanding the material parameters to each spatial location.

    Returns:
        z: Tensor of shape (B, N, N, 5).
    """
    B, N, _, _ = R.shape
    # Expand material_params: (B, 3) -> (B, 1, 1, 3) -> (B, N, N, 3)
    material_expanded = material_params.view(B, 1, 1, 3).expand(B, N, N, 3)
    # Concatenate along the last dimension.
    z = torch.cat([R, material_expanded], dim=-1)
    return z

def compute_gradient(u, dx, dy):
    """
    Compute spatial gradients using central finite differences with replication padding.
    u: Tensor of shape (B, N_x, N_y, C).
    Returns:
      du_dx, du_dy: both of shape (B, N_x, N_y, C).
    """
    B, Nx, Ny, C = u.shape
    # Pad with replication on spatial dimensions
    u_pad = F.pad(u, (0, 0, 1, 1, 1, 1), mode='replicate')  # (B, Nx+2, Ny+2, C)
    du_dx = (u_pad[:, 2:, 1:-1, :] - u_pad[:, :-2, 1:-1, :]) / (2 * dx)
    du_dy = (u_pad[:, 1:-1, 2:, :] - u_pad[:, 1:-1, :-2, :]) / (2 * dy)
    return du_dx, du_dy

def compute_strain(u, dx, dy):
    """
    Compute the linear strain field from displacement u.
    Returns epsilon_xx, epsilon_yy, epsilon_xy (each of shape (B, N_x, N_y)).
    """
    du_dx, du_dy = compute_gradient(u, dx, dy)
    epsilon_xx = du_dx[..., 0]            # d(u_x)/dx
    epsilon_yy = du_dy[..., 1]            # d(u_y)/dy
    epsilon_xy = 0.5 * (du_dx[..., 1] + du_dy[..., 0])
    return epsilon_xx, epsilon_yy, epsilon_xy

def compute_stress_from_strain(epsilon_xx, epsilon_yy, epsilon_xy, material_params):
    """
    Compute stress field under plane stress assumption using strain.

    Material parameters: tensor of shape (B, 3), where each row is [lambda, mu, nu].
    Strain fields: tensors (epsilon_xx, epsilon_yy, epsilon_xy) of shape (B, ...).

    Returns:
        sigma_xx, sigma_yy, sigma_xy: tensors of the same shape as the strain fields.

    Using plane stress relations:
      sigma_xx = E/(1 - nu**2) * (epsilon_xx + nu * epsilon_yy)
      sigma_yy = E/(1 - nu**2) * (epsilon_yy + nu * epsilon_xx)
      sigma_xy = E/(1 + nu) * epsilon_xy
    with
      E = mu * (3 * lambda + 2 * mu) / (lambda + mu)
    """
    # Extract parameters and reshape for broadcasting.
    lamda = material_params[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    mu    = material_params[:, 1].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
    nu    = material_params[:, 2].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)

    # Compute Young's modulus E.
    E = mu * (3 * lamda + 2 * mu) / (lamda + mu)

    # Compute stress components under plane stress.
    sigma_xx = E / (1 - nu**2) * (epsilon_xx + nu * epsilon_yy)
    sigma_yy = E / (1 - nu**2) * (epsilon_yy + nu * epsilon_xx)
    sigma_xy = E / (1 + nu) * epsilon_xy  # Note: gamma_xy = 2 * epsilon_xy, so alternatively: E/(2*(1+nu))*2*epsilon_xy

    return sigma_xx, sigma_yy, sigma_xy

def compute_divergence_of_stress(sigma_xx, sigma_yy, sigma_xy, dx, dy):
    """
    Compute divergence of the stress tensor on a given grid.
    Inputs:
      sigma_xx, sigma_yy, sigma_xy: each of shape (B, M, N) where M and N are spatial dims.
    Returns:
      divergence: Tensor of shape (B, M, N, 2) where last dimension holds [div_x, div_y].
    We use central differences with replication padding.
    """
    B, M, N = sigma_xx.shape

    def grad_scalar(f, dx, dy):
        # f: (B, M, N) -> add channel dimension and pad
        f = f.unsqueeze(-1)  # shape: (B, M, N, 1)
        f_pad = F.pad(f, (0, 0, 1, 1, 1, 1), mode='replicate').squeeze(-1)  # (B, M+2, N+2)
        df_dx = (f_pad[:, 2:, 1:-1] - f_pad[:, :-2, 1:-1]) / (2 * dx)
        df_dy = (f_pad[:, 1:-1, 2:] - f_pad[:, 1:-1, :-2]) / (2 * dy)
        return df_dx, df_dy

    d_sigma_xx_dx, _ = grad_scalar(sigma_xx, dx, dy)
    d_sigma_xy_dx, _ = grad_scalar(sigma_xy, dx, dy)
    _, d_sigma_xy_dy = grad_scalar(sigma_xy, dx, dy)
    _, d_sigma_yy_dy = grad_scalar(sigma_yy, dx, dy)

    div_x = d_sigma_xx_dx + d_sigma_xy_dy
    div_y = d_sigma_xy_dx + d_sigma_yy_dy
    divergence = torch.stack([div_x, div_y], dim=-1)  # (B, M, N, 2)
    return divergence

def extend_residual(R_int, pad_size=2):
    """
    Extend the residual computed on the interior grid (shape: (B, H, W, 2))
    back to the full grid shape by replicating the boundaries.

    Parameters:
      R_int: Tensor of shape (B, H, W, 2), where H = N_x - 4, W = N_y - 4.
      pad_size: Number of layers to pad on each side (default is 2).

    Returns:
      R_full: Tensor of shape (B, H+2*pad_size, W+2*pad_size, 2).
    """
    # Permute to channel-first: (B, 2, H, W)
    R_int_cf = R_int.permute(0, 3, 1, 2)
    # Apply padding: pad format for 4D tensor is (pad_left, pad_right, pad_top, pad_bottom)
    R_full_cf = F.pad(R_int_cf, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    # Permute back to channel-last: (B, H+2*pad_size, W+2*pad_size, 2)
    R_full = R_full_cf.permute(0, 2, 3, 1)
    return R_full

def fixed_point_update(u, alpha, R):
    """
    Fixed-point update: u_new = u - alpha * R.

    Inputs:
      u: Tensor of shape (B, Nx, Ny, 2) -- displacement field.
      alpha: Tensor of shape (B, Nx, Ny, 2, 2) -- per-grid-point update operator.
      R: Tensor of shape (B, Nx, Ny, 2) -- extended residual (divergence of stress).

    The multiplication alpha * R is done via matrix multiplication per grid point.
    """
    # Unsqueeze R to shape (B, Nx, Ny, 2, 1) so we can apply a 2x2 matrix multiplication.
    R_unsq = R.unsqueeze(-1)  # (B, Nx, Ny, 2, 1)
    # Multiply: result has shape (B, Nx, Ny, 2, 1)
    update = torch.matmul(alpha, R_unsq).squeeze(-1)  # (B, Nx, Ny, 2)

    # Fixed-point update: subtract the scaled residual.
    u_new = u - update
    return u_new

def apply_boundary_mask(u, u_bottom, u_top):
    """
    Apply boundary correction to ensure the bottom and top y-boundaries are fixed.

    Inputs:
      u: Tensor of shape (B, Nx, Ny, 2) -- updated displacement field.
      u_bottom: Tensor of shape (B, Nx, 2) -- prescribed displacement at y-bottom.
                (For example, zeros.)
      u_top: Tensor of shape (B, Nx, 2) -- prescribed displacement at y-top.
             This can be an array of values.

    The function forces:
      u[:, :, 0, :] = u_bottom  (bottom boundary)
      u[:, :, -1, :] = u_top    (top boundary)

    Returns:
      u_corrected: Tensor of shape (B, Nx, Ny, 2) with boundaries enforced.
    """
    # Create a copy (or work in-place if desired)
    u_corrected = u.clone()

    # Here, we assume the second spatial dimension (index 2) is y.
    # Enforce bottom boundary (y index 0)
    u_corrected[:, :, 0, :] = u_bottom  # u_bottom should be shape (B, Nx, 2)
    # Enforce top boundary (y index Ny-1)
    u_corrected[:, :, -1, :] = u_top
    return u_corrected

def fixed_point_iteration(u, material_params, u_top, u_bottom, dx, dy, model):
    """
    Compute f(u, m, u_top, u_bottom, dx, dy) = u - alpha(u, m)*R(u)
    where:
      - R(u) is computed from u by:
          (a) computing full-domain strain,
          (b) cropping to interior (remove one layer from each side),
          (c) computing stress from interior strain,
          (d) computing divergence on the interior,
          (e) extending R back to full grid.
      - Instead of computing alpha via mlp_alpha(u, material_params),
        we build z = [R_full, material_params] (shape: (B, N, N, 5)) and then compute
        new_u = model(u, z) where model is an instance of FixedPointResNet.
      - Finally, a boundary mask is applied using u_top and u_bottom.

    Args:
        u: Displacement field, shape (B, Nx, Ny, 2)
        material_params: Tensor of shape (3,) or (B, 3)
        u_top: Tensor of shape (B, Nx, 2)
        u_bottom: Tensor of shape (B, Nx, 2)
        dx, dy: Scalars.
        model: An instance of FixedPointResNet (which takes u and z as inputs)

    Returns:
        u_new: Updated displacement field, shape (B, Nx, Ny, 2)
    """
    device = u.device  # Ensure all variables use the same device as u
    B, Nx, Ny, _ = u.shape

    # Ensure material_params is on the correct device.
    material_params = material_params.to(device)
    # Also ensure u_top and u_bottom are on the correct device.
    u_top = u_top.to(device)
    u_bottom = u_bottom.to(device)

    # 1. Compute full-domain strain
    eps_xx, eps_yy, eps_xy = compute_strain(u, dx, dy)

    # 2. Crop strain to interior: remove one layer from each side.
    eps_xx_int = eps_xx[:, 1:-1, 1:-1]  # (B, Nx-2, Ny-2)
    eps_yy_int = eps_yy[:, 1:-1, 1:-1]
    eps_xy_int = eps_xy[:, 1:-1, 1:-1]

    # 3. Compute stress from interior strain (using plane stress assumptions)
    sigma_xx_int, sigma_yy_int, sigma_xy_int = compute_stress_from_strain(eps_xx_int, eps_yy_int, eps_xy_int, material_params)

    # 4. Compute divergence of interior stress
    R_int = compute_divergence_of_stress(sigma_xx_int, sigma_yy_int, sigma_xy_int, dx, dy)  # (B, Nx-2, Ny-2, 2)

    # 5. Extend R_int to full grid shape: (B, Nx, Ny, 2)
    R_full = extend_residual(R_int, pad_size=1)
    R_full = R_full.to(device)

    # 6. Build z by concatenating R_full and material_params.
    #    z will have shape (B, Nx, Ny, 5)
    z = build_z(R_full, material_params)
    z = z.to(device)

    # 7. Run the FixedPointResNet model to compute new u.
    u_new = model(u, z)  # new_u shape: (B, Nx, Ny, 2)

    # 8. Apply boundary mask using u_top and u_bottom.
    u_new = apply_boundary_mask(u_new, u_bottom, u_top)
    return u_new


def compute_global_alpha_for_sample(delta_u_sample, delta_R_sample, epsilon=1e-8):
    """
    For one sample (with spatial dimensions flattened to M points),
    solve the least-squares problem to find the 2x2 matrix alpha that minimizes:
        || delta_u + alpha * delta_R ||^2,
    i.e., we want delta_u = -alpha * delta_R.

    delta_u_sample: Tensor of shape (M, 2)
    delta_R_sample: Tensor of shape (M, 2)

    Returns:
      alpha_sample: Tensor of shape (2, 2)
    """
    # We'll solve two independent least-squares problems:
    # For the first row of alpha:
    #   a * (delta_R_x) + b * (delta_R_y) = - delta_u_x
    X = delta_R_sample  # shape (M, 2)
    y1 = -delta_u_sample[:, 0]  # shape (M,)
    # Solve for [a, b] (add an extra dimension for lstsq)
    sol1 = torch.linalg.lstsq(X, y1.unsqueeze(1)).solution.squeeze(1)

    # For the second row of alpha:
    #   c * (delta_R_x) + d * (delta_R_y) = - delta_u_y
    y2 = -delta_u_sample[:, 1]
    sol2 = torch.linalg.lstsq(X, y2.unsqueeze(1)).solution.squeeze(1)

    # Combine into a 2x2 matrix:
    # [ [a, b],
    #   [c, d] ]
    alpha_sample = torch.stack([sol1, sol2], dim=0)
    return alpha_sample

def generate_training_data(dx, dy, material_params,
                           B=32, Nx=32, Ny=32, delta=1e-4):
    """
    Generate (u, R, alpha) training data for learning an optimal step matrix alpha.

    For each random displacement field u (shape (B, Nx, Ny, 2)),
    we compute a small random perturbation delta_u and then the corresponding change in the residual field:
        delta_R = R(u + delta_u) - R(u).
    Then, for each sample in the batch, we solve the least-squares problem
        delta u = -alpha * delta R
    over all spatial points (flattening Nx x Ny to M points) to obtain a global 2x2 matrix alpha.

    Returns:
      u: Tensor of shape (B, Nx, Ny, 2)
      R: Tensor of shape (B, Nx, Ny, 2)
      alpha: Tensor of shape (B, 2, 2) (one global 2x2 matrix per sample)
    """
    device = material_params.device
    dtype = material_params.dtype

    # 1. Generate a random displacement field u: shape (B, Nx, Ny, 2)
    u = torch.rand(B, Nx, Ny, 2, device=device, dtype=dtype)
    delta_u = delta * torch.randn_like(u)  # small random perturbation

    # 2. Define a function to compute R(u)
    def compute_R(u_tensor):
        u_tensor.requires_grad_(True)
        eps_xx, eps_yy, eps_xy = compute_strain(u_tensor, dx, dy)
        eps_xx_int = eps_xx[:, 1:-1, 1:-1]
        eps_yy_int = eps_yy[:, 1:-1, 1:-1]
        eps_xy_int = eps_xy[:, 1:-1, 1:-1]
        sigma_xx_int, sigma_yy_int, sigma_xy_int = compute_stress_from_strain(
            eps_xx_int, eps_yy_int, eps_xy_int, material_params
        )
        R_int = compute_divergence_of_stress(sigma_xx_int, sigma_yy_int, sigma_xy_int, dx, dy)
        R_full = extend_residual(R_int, pad_size=1)  # shape: (B, Nx, Ny, 2)
        return R_full

    # 3. Compute residuals R at u and at the perturbed field
    R = compute_R(u)                          # shape (B, Nx, Ny, 2)
    R_perturbed = compute_R(u + delta_u)        # shape (B, Nx, Ny, 2)
    delta_R = R_perturbed - R                   # shape (B, Nx, Ny, 2)

    # 4. For each sample in the batch, compute a global alpha using the least-squares approach.
    alpha_list = []
    for i in range(B):
        # Flatten the spatial dimensions (Nx, Ny) into M points.
        delta_u_sample = delta_u[i].view(-1, 2)   # shape (Nx*Ny, 2)
        delta_R_sample = delta_R[i].view(-1, 2)   # shape (Nx*Ny, 2)
        alpha_sample = compute_global_alpha_for_sample(delta_u_sample, delta_R_sample)
        alpha_list.append(alpha_sample)
    alpha = torch.stack(alpha_list, dim=0)  # shape (B, 2, 2)

    return u, R, alpha

if __name__ == "__main__":
  # Grid and batch settings
  data = np.load("/home/doelz-admin/projects/chebysev_deq/elastic_training_data.npz")

  # === Extract displacement ===
  u_x = data["u_x"]  # shape: (Nx, Ny)
  u_y = data["u_y"]  # shape: (Nx, Ny)

  # Stack into u: shape → (1, Nx, Ny, 2)
  u = np.stack([u_x, u_y], axis=-1)  # shape: (Nx, Ny, 2)
  u = u[np.newaxis, ...]             # shape: (1, Nx, Ny, 2)

  # === Extract strain (optional, if you want to use for supervised loss) ===
  epsilon = np.stack([
      data["epsilon_xx"],
      data["epsilon_yy"],
      data["epsilon_xy"]
  ], axis=-1)  # shape: (Nx, Ny, 3)
  epsilon = epsilon[np.newaxis, ...]  # (1, Nx, Ny, 3)

  # === Extract stress (optional) ===
  sigma = np.stack([
      data["sigma_xx"],
      data["sigma_yy"],
      data["sigma_xy"]
  ], axis=-1)  # shape: (Nx, Ny, 3)
  sigma = sigma[np.newaxis, ...]

  # === Extract shape ===
  B, Nx, Ny, _ = u.shape
  print(f"Loaded shape: B={B}, Nx={Nx}, Ny={Ny}")

  # === Convert to torch.Tensor ===
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  u_tensor = torch.tensor(u, dtype=torch.float32, device=device, requires_grad=True)

  dx = 1.0 / (Nx - 1)
  dy = 1.0 / (Ny - 1)
  u_init = torch.rand(B, Nx, Ny, 2).to(device)
  # Create initial displacement field u: shape (B, Nx, Ny, 2)
  material_params = torch.tensor([[100.0, 80.0, 0.3]]).expand(B, -1).to(device)
  # Prescribed top and bottom displacement values
  u_bottom = torch.zeros(B, Nx, 2, device=device)        # u = 0 at y=0
  u_top = torch.zeros(B, Nx, 2, device=device)           # u_y = 0.1 at y=1
  u_top[..., 1] = 0.1
  model = FixedPointResNet(u_in_channels=2, z_in_channels=5, hidden_channels=32, num_groups=8).to(device)
  f_fixed = lambda u: fixed_point_iteration(
        u,
        material_params=material_params,
        u_top=u_top,
        u_bottom=u_bottom,
        dx=dx,
        dy=dy,
        model=model
    )

  # Run one fixed-point update
  deq = get_deq(f_solver='broyden', f_max_iter=200, f_tol=1e-9)
  u_out, info = deq(f_fixed, u_init)
  print(u_out)
  f_abs_trace = info['abs_trace']
  f_abs_trace = f_abs_trace.mean(dim=0)[1:]
  iterations = np.arange(len(f_abs_trace))+1
  plt.plot(iterations, f_abs_trace.cpu(), 'o-', color='#4f96b8', markersize=8, linewidth=2, label='$f(z) = \cos(z)$')
  plt.savefig("fixed_iter.png")

  # Create a dummy residual field R of shape (B, N, N, 2)


  # alpha_field =alphat.view(32, 1, 1, 2, 2).expand(B, Nx, Ny, 2, 2)
  # print(alpha_field.shape)






