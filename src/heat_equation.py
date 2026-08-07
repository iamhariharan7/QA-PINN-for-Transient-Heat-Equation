import numpy as np

class HeatEquationDomain:
    def __init__(self, L=1.0, T=1.0, Nx=30, Nt=30, alpha=0.01, dataset_dict=None, props=None):
        if dataset_dict is not None:
            self.x = np.array(dataset_dict['x'], dtype=float)
            self.t = np.array(dataset_dict['t'], dtype=float)
            self.U_exact = np.array(dataset_dict['U'], dtype=float)
            self.Nx = len(self.x)
            self.Nt = len(self.t)
            self.L = float(self.x.max() - self.x.min()) if self.Nx > 1 else L
            self.T_max = float(self.t.max() - self.t.min()) if self.Nt > 1 else T
        else:
            self.L = float(L)
            self.T_max = float(T)
            self.Nx = int(Nx)
            self.Nt = int(Nt)
            self.x = np.linspace(0, self.L, self.Nx)
            self.t = np.linspace(0, self.T_max, self.Nt)
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

        self.X, self.T = np.meshgrid(self.x, self.t)
        self.dx = float(self.x[1] - self.x[0]) if self.Nx > 1 else 1.0
        self.dt = float(self.t[1] - self.t[0]) if self.Nt > 1 else 1.0

    def get_grid(self):
        return self.X, self.T, self.x, self.t

    def initial_condition(self, x_eval):
        """Returns initial temperature distribution u(x, 0) from dataset or analytical fallback."""
        if self.U_exact is not None:
            u0 = self.U_exact[0]
            if hasattr(x_eval, 'detach'): # torch tensor
                x_np = x_eval.detach().cpu().numpy()
                interp = np.interp(x_np, self.x, u0)
                import torch
                return torch.from_numpy(interp).to(dtype=x_eval.dtype, device=x_eval.device)
            else:
                return np.interp(x_eval, self.x, u0)
        else:
            if hasattr(x_eval, 'sin'):
                import torch
                return torch.sin(np.pi * x_eval / self.L)
            return np.sin(np.pi * x_eval / self.L)

    def boundary_conditions(self):
        """Returns left and right boundary conditions from dataset or zero fallback."""
        if self.U_exact is not None:
            left_bc = self.U_exact[:, 0]
            right_bc = self.U_exact[:, -1]
            return left_bc, right_bc
        return 0.0, 0.0
