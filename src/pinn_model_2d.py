import torch
import torch.nn as nn
import numpy as np
import time
import os

class PINN2D(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),  nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_star: torch.Tensor, y_star: torch.Tensor, t_star: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_star, y_star, t_star], dim=1)
        return self.net(inp)

def solve_pinn_2d(domain2d, config: dict, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)

    start_time = time.time()

    n_train = config.get("n_train", 800)
    epochs  = config.get("epochs",  2000)
    lr      = config.get("lr",      1e-3)

    Lx, Ly = domain2d.Lx, domain2d.Ly
    T_max  = domain2d.T_max
    
    u_min = float(domain2d.U_exact.min()) if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None else 300.0
    u_max = float(domain2d.U_exact.max()) if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    
    alpha_star_x = domain2d.alpha * T_max / (Lx**2)
    alpha_star_y = domain2d.alpha * T_max / (Ly**2)

    x_phys_star = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)
    y_phys_star = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)
    t_phys_star = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)

    x_ic_star   = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)
    y_ic_star   = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)
    t_bc_star   = torch.FloatTensor(n_train, 1).uniform_(0.0, 1.0)

    model     = PINN2D(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Random Collocation Subsampling for Speed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_pts = x_phys_star.shape[0]
    n_colloc = min(total_pts, 5000)
    idx = torch.randperm(total_pts)[:n_colloc]
    x_phys_star = x_phys_star[idx].to(device)
    t_phys_star = t_phys_star[idx].to(device)
    x_ic_star = x_ic_star.to(device)
    t_bc_star = t_bc_star.to(device)
    x0_star = x0_star.to(device) if 'x0_star' in locals() else torch.zeros_like(t_bc_star).to(device)
    x1_star = x1_star.to(device) if 'x1_star' in locals() else torch.ones_like(t_bc_star).to(device)
    if 'y_phys_star' in locals(): y_phys_star = y_phys_star[idx].to(device)
    if 'z_phys_star' in locals(): z_phys_star = z_phys_star[idx].to(device)
    if 'y_ic_star' in locals(): y_ic_star = y_ic_star.to(device)
    if 'z_ic_star' in locals(): z_ic_star = z_ic_star.to(device)

    def dynamic_loss():
        x_p = x_phys_star.requires_grad_(True)
        y_p = y_phys_star.requires_grad_(True)
        t_p = t_phys_star.requires_grad_(True)

        u_star = model(x_p, y_p, t_p)
        ones = torch.ones_like(u_star)

        u_t  = torch.autograd.grad(u_star, t_p, ones, create_graph=True)[0]
        u_x  = torch.autograd.grad(u_star, x_p, ones, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_p, ones, create_graph=True)[0]
        u_y  = torch.autograd.grad(u_star, y_p, ones, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_p, ones, create_graph=True)[0]

        pde_residual = u_t - (alpha_star_x * u_xx + alpha_star_y * u_yy)
        loss_pde = torch.mean(pde_residual ** 2)

        t_zero_star = torch.zeros_like(x_ic_star)
        u_ic_star = model(x_ic_star, y_ic_star, t_zero_star)
        
        ic_phys_numpy = domain2d.initial_condition((x_ic_star * Lx).detach().cpu().numpy(), (y_ic_star * Ly).detach().cpu().numpy())
        u_ic_exact_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
        if u_ic_exact_phys.ndim == 1:
            u_ic_exact_phys = u_ic_exact_phys.unsqueeze(1)
        u_ic_exact_star = (u_ic_exact_phys - u_min) / u_scale
        loss_ic = torch.mean((u_ic_star - u_ic_exact_star) ** 2)

        x0_star = torch.zeros_like(t_bc_star)
        x1_star = torch.ones_like(t_bc_star)
        y0_star = torch.zeros_like(t_bc_star)
        y1_star = torch.ones_like(t_bc_star)
        y_rand_star = torch.FloatTensor(t_bc_star.size()).uniform_(0.0, 1.0).to(device)
        x_rand_star = torch.FloatTensor(t_bc_star.size()).uniform_(0.0, 1.0).to(device)

        bc_x0_phys = domain2d.initial_condition(np.zeros_like(y_rand_star.cpu().numpy()) * Lx, (y_rand_star * Ly).detach().cpu().numpy())
        bc_x1_phys = domain2d.initial_condition(np.ones_like(y_rand_star.cpu().numpy()) * Lx, (y_rand_star * Ly).detach().cpu().numpy())
        bc_y0_phys = domain2d.initial_condition((x_rand_star * Lx).detach().cpu().numpy(), np.zeros_like(x_rand_star.cpu().numpy()) * Ly)
        bc_y1_phys = domain2d.initial_condition((x_rand_star * Lx).detach().cpu().numpy(), np.ones_like(x_rand_star.cpu().numpy()) * Ly)
        
        bc_x0_val = (torch.tensor(bc_x0_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_x1_val = (torch.tensor(bc_x1_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_y0_val = (torch.tensor(bc_y0_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_y1_val = (torch.tensor(bc_y1_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        
        if bc_x0_val.ndim == 1: bc_x0_val = bc_x0_val.unsqueeze(1)
        if bc_x1_val.ndim == 1: bc_x1_val = bc_x1_val.unsqueeze(1)
        if bc_y0_val.ndim == 1: bc_y0_val = bc_y0_val.unsqueeze(1)
        if bc_y1_val.ndim == 1: bc_y1_val = bc_y1_val.unsqueeze(1)

        loss_bc = torch.mean((model(x0_star, y_rand_star, t_bc_star) - bc_x0_val)**2) + \
                  torch.mean((model(x1_star, y_rand_star, t_bc_star) - bc_x1_val)**2) + \
                  torch.mean((model(x_rand_star, y0_star, t_bc_star) - bc_y0_val)**2) + \
                  torch.mean((model(x_rand_star, y1_star, t_bc_star) - bc_y1_val)**2)

        return loss_pde + 100.0 * loss_ic + 100.0 * loss_bc

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(10, epochs//20), factor=0.5, min_lr=1e-6)

    losses = []
    from src.metrics import count_parameters
    p_count = count_parameters(model)
    print(f"  -> Working with {p_count:,} active neurons/parameters.")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = dynamic_loss()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        scheduler.step(current_loss)
        losses.append(current_loss)
        if current_loss < 1e-3:
            print(f'  [+] Early stopping at epoch {epoch+1} (Loss < 1e-3)')
            break
        
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4e} - LR: {current_lr:.2e}")

    pinn2d_time = time.time() - start_time

    X3, Y3, T3 = domain2d.get_xyt_grid()
    x_star_flat = torch.FloatTensor((X3 / Lx).reshape(-1, 1)).to(device)
    y_star_flat = torch.FloatTensor((Y3 / Ly).reshape(-1, 1)).to(device)
    t_star_flat = torch.FloatTensor((T3 / T_max).reshape(-1, 1)).to(device)

    model.eval()
    inf_start = time.time()
    with torch.no_grad():
        u_star_pred = model(x_star_flat, y_star_flat, t_star_flat).cpu().numpy()
        u_pred = u_min + u_scale * u_star_pred

    U_pinn2d = u_pred.reshape(domain2d.Nt, domain2d.Ny, domain2d.Nx)

    inference_time = time.time() - inf_start
    if output_dir:
        target_dir = os.path.join(output_dir, "PINN", "2D")
        os.makedirs(target_dir, exist_ok=True)
        np.savez(
            os.path.join(target_dir, "pinn_temperature_2d.npz"),
            U=U_pinn2d,
            x=domain2d.x,
            y=domain2d.y,
            t=domain2d.t,
        )

    return U_pinn2d, pinn2d_time, inference_time, losses, model
