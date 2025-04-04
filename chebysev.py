import torch

def uniform_to_chebyshev_coords(
    coords: torch.Tensor,
    N: int,
    domain: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized mapping of uniform coordinates to Chebyshev node positions.

    Args:
        coords: (num_points, D) tensor, coordinates in a box domain.
        N: int, number of Chebyshev nodes per axis (same for all dimensions).
        domain: (2, D) tensor, where domain[0] = min, domain[1] = max for each dim.

    Returns:
        cheb_coords: (num_points, D) tensor of mapped Chebyshev node locations.
    """
    # Normalize from [a_d, b_d] → [-1, 1]
    a = domain[0]  # shape (D,)
    b = domain[1]  # shape (D,)
    coords_norm = 2 * (coords - a) / (b - a) - 1  # shape (N_pts, D), now in [-1, 1]

    # Compute pseudo Chebyshev indices
    k = 0.5 * (N - 1) * (coords_norm + 1)  # shape (N_pts, D)

    # Apply Chebyshev transform
    cheb = torch.cos((2 * k + 1) * torch.pi / (2 * N))  # shape (N_pts, D)

    return cheb


def uniform_to_chebyshev_lobatto_coords(
    coords: torch.Tensor,
    N: int,
    domain: torch.Tensor
) -> torch.Tensor:
    """
    Map a uniform grid in a box domain to Chebyshev–Gauss–Lobatto (CGL) nodes.

    CGL nodes are defined by:  x_i = cos(i*pi / N),  i=0,...,N

    Args:
        coords:  (num_points, D) tensor, uniform coordinates in [a,b]^D.
        N:       int, number of intervals in the GLL scheme (so you have N+1 nodes).
        domain:  (2, D) tensor, where domain[0] = a (min per dim), domain[1] = b (max per dim).

    Returns:
        cheb_lobatto: (num_points, D) tensor of GLL coordinates in [-1,1]^D.
    """
    # domain[0], domain[1] are shape (D,). We broadcast over coords shape (num_points, D)
    a = domain[0]  # shape (D,)
    b = domain[1]  # shape (D,)

    # 1) Normalize to [-1,1]
    coords_norm = 2.0 * (coords - a) / (b - a) - 1.0  # shape (num_points, D)

    # 2) Continuous index i in [0, N]
    i = 0.5 * N * (coords_norm + 1.0)  # shape (num_points, D)

    # 3) Compute GLL coordinate
    x_gll = torch.cos(i * torch.pi / N)  # shape (num_points, D)

    return x_gll

def chebyshev_gll_1d(N: int, device=None, dtype=None):
    """
    Generate the 1D Chebyshev–Gauss–Lobatto (CGL) nodes in [-1,1]
    and the corresponding differentiation matrix of size (N+1)×(N+1).

    Args:
        N (int): Number of intervals. The grid has N+1 points i=0..N.
        device, dtype: Optional PyTorch device and dtype.

    Returns:
        x (torch.Tensor): shape (N+1,), the CGL nodes in [-1,1].
        D (torch.Tensor): shape (N+1, N+1), the differentiation matrix.
    """
    # i from 0..N
    i_idx = torch.arange(N+1, device=device, dtype=dtype)
    # x_i = cos(i*pi/N)
    x = torch.cos(torch.pi * i_idx / N)

    # c_i: c_0 = c_N = 2, else 1
    c = torch.ones(N+1, device=device, dtype=dtype)
    c[0]  = 2.0
    c[-1] = 2.0

    # Build 2D grids for broadcasting
    # shape => (N+1,1) vs (1,N+1) => results in (N+1,N+1)
    i_row = i_idx.view(N+1, 1)
    j_col = i_idx.view(1, N+1)

    x_row = x.view(N+1, 1)  # shape (N+1,1)
    x_col = x.view(1, N+1)  # shape (1,N+1)

    c_row = c.view(N+1, 1)
    c_col = c.view(1, N+1)

    # Off-diagonal formula:
    # D_ij = (c_i/c_j)*((-1)^(i+j)) / (x_i - x_j)  for i != j
    # We'll build a (N+1,N+1) tensor with that, then fix diagonal afterwards.

    # (x_j - x_i) for broadcasting
    denom = x_row - x_col # shape (N+1, N+1)
    c_ratio = c_row / c_col  # shape (N+1, N+1)

    # sign = (-1)^(i+j)
    # We can compute that by checking parity of (i+j).
    # Let's do a quick approach with modulo:
    parity = (i_row + j_col) % 2  # 0 or 1
    sign = (1.0 - 2.0 * parity)   # if parity=0 => sign=1, else sign=-1

    # Initialize D
    D = torch.zeros(N+1, N+1, device=device, dtype=dtype)

    # We'll fill off-diagonal entries with the formula
    off_diag_mask = (i_row != j_col)
    D[off_diag_mask] = (
        c_ratio[off_diag_mask] * sign[off_diag_mask] / denom[off_diag_mask]
    )

    # Diagonal: D_ii = - sum_{j != i} D_{ij}
    # sum along each row i
    diag_vals = -torch.sum(D, dim=1)
    # Fill diagonal
    D.fill_diagonal_(0.0)
    D[torch.arange(N+1), torch.arange(N+1)] = diag_vals

    return x, D

def differentiate_1d(f_values: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Given function samples f_values at Chebyshev–Gauss–Lobatto nodes,
    apply the differentiation matrix D to approximate f'(x).

    Args:
        f_values (torch.Tensor): shape (N+1,) function values at the CGL nodes
        D (torch.Tensor): shape (N+1, N+1) the differentiation matrix

    Returns:
        df (torch.Tensor): shape (N+1,) approximate derivative at each node
    """
    return D @ f_values

def example_usage():
    # Suppose N=8 => 9 CGL points
    N = 8
    x, D = chebyshev_gll_1d(N)

    # Suppose f(x) = sin(pi*x)
    # Evaluate at the CGL nodes
    f_values = torch.sin(torch.pi * x)

    # Approx derivative
    df_approx = differentiate_1d(f_values, D)

    # Exact derivative: df/dx = pi*cos(pi*x)
    df_exact = torch.pi * torch.cos(torch.pi * x)

    # Print
    print("CGL nodes x:\n", x)
    print("f values:\n", f_values)
    print("Approx derivative:\n", df_approx)
    print("Exact derivative:\n", df_exact)
    err = torch.norm(df_approx - df_exact, p=float('inf'))
    print(f"Max error: {err.item():.4e}")

def kron3(a, b, c):
    """Triple Kronecker product: a ⊗ b ⊗ c"""
    return torch.kron(torch.kron(a, b), c)

def chebyshev_tensor_diff_2d(D: torch.Tensor):
    """
    Construct 2D Chebyshev differentiation matrices D_x and D_y
    from 1D differentiation matrix D using Kronecker product.

    Args:
        D (Tensor): (N+1, N+1) 1D Chebyshev differentiation matrix

    Returns:
        D_x, D_y: each of shape ((N+1)^2, (N+1)^2)
    """
    I = torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
    D_x = torch.kron(D, I)  # D ⊗ I
    D_y = torch.kron(I, D)  # I ⊗ D
    return D_x, D_y

def chebyshev_tensor_diff_3d(D: torch.Tensor):
    """
    Construct 3D Chebyshev differentiation matrices D_x, D_y, D_z
    from 1D differentiation matrix D using Kronecker product.

    Returns:
        D_x, D_y, D_z: each of shape ((N+1)^3, (N+1)^3)
    """
    I = torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
    D_x = kron3(D, I, I)  # D ⊗ I ⊗ I
    D_y = kron3(I, D, I)  # I ⊗ D ⊗ I
    D_z = kron3(I, I, D)  # I ⊗ I ⊗ D
    return D_x, D_y, D_z


if __name__ == "__main__":
 # Suppose we have a 2D uniform grid in [0,2] x [-1,3], 8x8
  x_vals = torch.linspace(0.0, 2.0, 8)
  y_vals = torch.linspace(-1.0, 3.0, 8)
  X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')  # shape (8,8)

  coords_2d = torch.stack([X.flatten(), Y.flatten()], dim=1)  # shape (64,2)

  # Domain: shape (2,2)
  domain_2d = torch.tensor([[0.0, -1.0],
                            [2.0,  3.0]])

  # Convert to Chebyshev–Gauss–Lobatto
  N = 8  # => nodes i=0..8
  cheb_lobatto = uniform_to_chebyshev_lobatto_coords(coords_2d, N=N, domain=domain_2d)
  print(cheb_lobatto.shape)  # (64, 2)
  print(cheb_lobatto[:8])    # first 8 points mapped
  example_usage()
  N = 8
  x_1d, D_1d = chebyshev_gll_1d(N, dtype=torch.float64)

  # Step 2: Create 2D grid
  X, Y = torch.meshgrid(x_1d, x_1d, indexing='ij')  # shape (N+1, N+1)

  # Step 3: Define test function
  f = torch.sin(torch.pi * X) * torch.cos(torch.pi * Y)        # shape (N+1, N+1)
  df_dx_true = torch.pi * torch.cos(torch.pi * X) * torch.cos(torch.pi * Y)
  df_dy_true = -torch.pi * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)

  # Step 4: Flatten f to vector (row-major flattening)
  f_flat = f.flatten()  # shape (N+1)*(N+1)

  # Step 5: Build 2D differentiation matrices
  D_x, D_y = chebyshev_tensor_diff_2d(D_1d)  # each (81, 81)

  # Step 6: Apply numerical differentiation
  df_dx_flat = D_x @ f_flat
  df_dy_flat = D_y @ f_flat

  # Step 7: Reshape back to (N+1, N+1)
  df_dx = df_dx_flat.view(N+1, N+1)
  df_dy = df_dy_flat.view(N+1, N+1)

  # Step 8: Compare with analytical results
  err_dx = torch.max(torch.abs(df_dx - df_dx_true))
  err_dy = torch.max(torch.abs(df_dy - df_dy_true))

  print(f"Max error in df/dx: {err_dx:.2e}")
  print(f"Max error in df/dy: {err_dy:.2e}")


