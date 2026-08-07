import numpy as np
from scipy.interpolate import RegularGridInterpolator

class HeatEquationDomain3D:
    def __init__(self, Lx=1.0, Ly=1.0, Lz=1.0, T=1.0, Nx=10, Ny=10, Nz=10, Nt=10, alpha=0.01, dataset_dict=None, props=None):
        if dataset_dict is not None:
            self.x = np.array(dataset_dict['x'], dtype=float)
            self.y = np.array(dataset_dict['y'], dtype=float)
            self.z = np.array(dataset_dict['z'], dtype=float)
            self.t = np.array(dataset_dict['t'], dtype=float)
            self.U_exact = np.array(dataset_dict['U'], dtype=float) # (Nt, Nx, Ny, Nz)
            self.Nx = len(self.x)
            self.Ny = len(self.y)
            self.Nz = len(self.z)
            self.Nt = len(self.t)
            self.Lx = float(self.x.max() - self.x.min()) if self.Nx > 1 else Lx
            self.Ly = float(self.y.max() - self.y.min()) if self.Ny > 1 else Ly
            self.Lz = float(self.z.max() - self.z.min()) if self.Nz > 1 else Lz
            self.T_max = float(self.t.max() - self.t.min()) if self.Nt > 1 else T
        else:
            self.Lx = float(Lx)
            self.Ly = float(Ly)
            self.Lz = float(Lz)
            self.T_max = float(T)
            self.Nx = int(Nx)
            self.Ny = int(Ny)
            self.Nz = int(Nz)
            self.Nt = int(Nt)
            self.x = np.linspace(0.0, self.Lx, self.Nx)
            self.y = np.linspace(0.0, self.Ly, self.Ny)
            self.z = np.linspace(0.0, self.Lz, self.Nz)
            self.t = np.linspace(0.0, self.T_max, self.Nt)
            self.U_exact = None

        if props is not None:
            self.alpha = float(props['alpha'])
            self.k = float(props['k'])
            self.rho = float(props['rho'])
            self.cp = float(props['cp'])
            self.material_name = props.get('name', 'Unknown')
        else:
            self.alpha = float(alpha)
            self.k = None
            self.rho = None
            self.cp = None
            self.material_name = 'Unknown'

        self.dx = float(self.x[1] - self.x[0]) if self.Nx > 1 else 1.0
        self.dy = float(self.y[1] - self.y[0]) if self.Ny > 1 else 1.0
        self.dz = float(self.z[1] - self.z[0]) if self.Nz > 1 else 1.0
        self.dt = float(self.t[1] - self.t[0]) if self.Nt > 1 else 1.0

        if self.U_exact is not None:
            self._ic_interp = RegularGridInterpolator((self.x, self.y, self.z), self.U_exact[0], bounds_error=False, fill_value=None)
        else:
            self._ic_interp = None

    def initial_condition(self, X, Y, Z):
        """Returns initial temperature distribution u(x, y, z, 0) from dataset or analytical fallback."""
        if self._ic_interp is not None:
            if hasattr(X, 'detach'):
                import torch
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()
                Z_np = Z.detach().cpu().numpy()
                pts = np.column_stack([X_np.ravel(), Y_np.ravel(), Z_np.ravel()])
                interp_vals = self._ic_interp(pts).reshape(X_np.shape)
                return torch.from_numpy(interp_vals).to(dtype=X.dtype, device=X.device)
            else:
                pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                return self._ic_interp(pts).reshape(X.shape)
        else:
            if hasattr(X, 'sin'):
                import torch
                return torch.sin(np.pi * X / self.Lx) * torch.sin(np.pi * Y / self.Ly) * torch.sin(np.pi * Z / self.Lz)
            return np.sin(np.pi * X / self.Lx) * np.sin(np.pi * Y / self.Ly) * np.sin(np.pi * Z / self.Lz)

    def boundary_conditions(self):
        return 0.0

    def get_xyz_grid(self):
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')

    def get_xyzt_grid(self):
        X3, Y3, Z3 = self.get_xyz_grid()
        X4 = X3[np.newaxis, :, :, :]
        Y4 = Y3[np.newaxis, :, :, :]
        Z4 = Z3[np.newaxis, :, :, :]
        T4 = self.t[:, np.newaxis, np.newaxis, np.newaxis]
        return np.broadcast_to(X4, (self.Nt, self.Nx, self.Ny, self.Nz)).copy(), \
               np.broadcast_to(Y4, (self.Nt, self.Nx, self.Ny, self.Nz)).copy(), \
               np.broadcast_to(Z4, (self.Nt, self.Nx, self.Ny, self.Nz)).copy(), \
               np.broadcast_to(T4, (self.Nt, self.Nx, self.Ny, self.Nz)).copy()
