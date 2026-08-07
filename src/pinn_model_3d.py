import torch
import torch.nn as nn
import numpy as np
import time
import os

class PINN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 1)
        )

    def forward(self, x_star, y_star, z_star, t_star):
        return self.net(torch.cat([x_star, y_star, z_star, t_star], dim=1))

def solve_pinn_3d(domain3d, config, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    
    model = PINN3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    
    n_train = config.get('n_train', 500)
    epochs = config.get('epochs', 1000)
    
    Lx, Ly, Lz, T_max = domain3d.Lx, domain3d.Ly, domain3d.Lz, domain3d.T_max
    u_min = float(domain3d.U_exact.min()) if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None else 300.0
    u_max = float(domain3d.U_exact.max()) if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    
    alpha_star_x = domain3d.alpha * T_max / (Lx**2)
    alpha_star_y = domain3d.alpha * T_max / (Ly**2)
    alpha_star_z = domain3d.alpha * T_max / (Lz**2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_pts = n_train
    x_phys_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    y_phys_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    z_phys_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    t_phys_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    
    x_ic_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    y_ic_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    z_ic_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    t_ic_star = torch.zeros(total_pts, 1).to(device)
    
    x_bc_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    y_bc_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    z_bc_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)
    t_bc_star = torch.FloatTensor(total_pts, 1).uniform_(0, 1).to(device)

    def dynamic_loss():
        x_p = x_phys_star.requires_grad_(True)
        y_p = y_phys_star.requires_grad_(True)
        z_p = z_phys_star.requires_grad_(True)
        t_p = t_phys_star.requires_grad_(True)

        u_star = model(x_p, y_p, z_p, t_p)
        ones = torch.ones_like(u_star)

        u_t  = torch.autograd.grad(u_star, t_p, ones, create_graph=True)[0]
        u_x  = torch.autograd.grad(u_star, x_p, ones, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_p, ones, create_graph=True)[0]
        u_y  = torch.autograd.grad(u_star, y_p, ones, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_p, ones, create_graph=True)[0]
        u_z  = torch.autograd.grad(u_star, z_p, ones, create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z_p, ones, create_graph=True)[0]

        pde_residual = u_t - (alpha_star_x * u_xx + alpha_star_y * u_yy + alpha_star_z * u_zz)
        loss_pde = torch.mean(pde_residual ** 2)

        u_ic_star_pred = model(x_ic_star, y_ic_star, z_ic_star, t_ic_star)
        ic_phys_numpy = domain3d.initial_condition((x_ic_star * Lx).detach().cpu().numpy(), (y_ic_star * Ly).detach().cpu().numpy(), (z_ic_star * Lz).detach().cpu().numpy())
        u_ic_exact_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
        if u_ic_exact_phys.ndim == 1:
            u_ic_exact_phys = u_ic_exact_phys.unsqueeze(1)
        u_ic_exact_star = (u_ic_exact_phys - u_min) / u_scale
        loss_ic = torch.mean((u_ic_star_pred - u_ic_exact_star) ** 2)

        x0_star = torch.zeros_like(t_bc_star)
        x1_star = torch.ones_like(t_bc_star)
        
        bc_x0_phys = domain3d.initial_condition(np.zeros_like(y_bc_star.cpu().numpy()) * Lx, (y_bc_star * Ly).detach().cpu().numpy(), (z_bc_star * Lz).detach().cpu().numpy())
        bc_x1_phys = domain3d.initial_condition(np.ones_like(y_bc_star.cpu().numpy()) * Lx, (y_bc_star * Ly).detach().cpu().numpy(), (z_bc_star * Lz).detach().cpu().numpy())
        bc_y0_phys = domain3d.initial_condition((x_bc_star * Lx).detach().cpu().numpy(), np.zeros_like(x_bc_star.cpu().numpy()) * Ly, (z_bc_star * Lz).detach().cpu().numpy())
        bc_y1_phys = domain3d.initial_condition((x_bc_star * Lx).detach().cpu().numpy(), np.ones_like(x_bc_star.cpu().numpy()) * Ly, (z_bc_star * Lz).detach().cpu().numpy())
        bc_z0_phys = domain3d.initial_condition((x_bc_star * Lx).detach().cpu().numpy(), (y_bc_star * Ly).detach().cpu().numpy(), np.zeros_like(z_bc_star.cpu().numpy()) * Lz)
        bc_z1_phys = domain3d.initial_condition((x_bc_star * Lx).detach().cpu().numpy(), (y_bc_star * Ly).detach().cpu().numpy(), np.ones_like(z_bc_star.cpu().numpy()) * Lz)
        
        bc_x0_val = (torch.tensor(bc_x0_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_x1_val = (torch.tensor(bc_x1_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_y0_val = (torch.tensor(bc_y0_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_y1_val = (torch.tensor(bc_y1_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_z0_val = (torch.tensor(bc_z0_phys, dtype=torch.float32, device=device) - u_min) / u_scale
        bc_z1_val = (torch.tensor(bc_z1_phys, dtype=torch.float32, device=device) - u_min) / u_scale

        if bc_x0_val.ndim == 1: bc_x0_val = bc_x0_val.unsqueeze(1)
        if bc_x1_val.ndim == 1: bc_x1_val = bc_x1_val.unsqueeze(1)
        if bc_y0_val.ndim == 1: bc_y0_val = bc_y0_val.unsqueeze(1)
        if bc_y1_val.ndim == 1: bc_y1_val = bc_y1_val.unsqueeze(1)
        if bc_z0_val.ndim == 1: bc_z0_val = bc_z0_val.unsqueeze(1)
        if bc_z1_val.ndim == 1: bc_z1_val = bc_z1_val.unsqueeze(1)

        loss_bc = torch.mean((model(x0_star, y_bc_star, z_bc_star, t_bc_star) - bc_x0_val)**2) + \
                  torch.mean((model(x1_star, y_bc_star, z_bc_star, t_bc_star) - bc_x1_val)**2) + \
                  torch.mean((model(x_bc_star, x0_star, z_bc_star, t_bc_star) - bc_y0_val)**2) + \
                  torch.mean((model(x_bc_star, x1_star, z_bc_star, t_bc_star) - bc_y1_val)**2) + \
                  torch.mean((model(x_bc_star, y_bc_star, x0_star, t_bc_star) - bc_z0_val)**2) + \
                  torch.mean((model(x_bc_star, y_bc_star, x1_star, t_bc_star) - bc_z1_val)**2)

        return loss_pde + 100.0 * loss_ic + 100.0 * loss_bc

    losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(10, epochs//20), factor=0.5, min_lr=1e-6)
    
    from src.metrics import count_parameters
    p_count = count_parameters(model)
    print(f"  -> Working with {p_count:,} active neurons/parameters.")

    # Adam Optimization Phase
    for epoch in range(int(epochs * 0.8)):
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
            print(f"Epoch {epoch+1}/{epochs} (Adam) - Loss: {current_loss:.4e} - LR: {current_lr:.2e}")

    # L-BFGS Optimization Phase
    lbfgs_opt = torch.optim.LBFGS(model.parameters(), max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    
    def closure():
        lbfgs_opt.zero_grad()
        loss = dynamic_loss()
        loss.backward()
        return loss

    for epoch in range(int(epochs * 0.8), epochs):
        loss = lbfgs_opt.step(closure)
        current_loss = loss.item()
        losses.append(current_loss)
        if current_loss < 1e-3:
            print(f'  [+] Early stopping at epoch {epoch+1} (Loss < 1e-3)')
            break
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} (L-BFGS) - Loss: {current_loss:.4e}")
    pinn_time = time.time() - start_time
    
    X4, Y4, Z4, T4 = domain3d.get_xyzt_grid()
    x_star_flat = torch.FloatTensor((X4 / Lx).reshape(-1, 1)).to(device)
    y_star_flat = torch.FloatTensor((Y4 / Ly).reshape(-1, 1)).to(device)
    z_star_flat = torch.FloatTensor((Z4 / Lz).reshape(-1, 1)).to(device)
    t_star_flat = torch.FloatTensor((T4 / T_max).reshape(-1, 1)).to(device)
    
    inf_start = time.time()
    with torch.no_grad():
        u_star_pred = model(x_star_flat, y_star_flat, z_star_flat, t_star_flat).cpu().numpy()
        u_pred = u_min + u_scale * u_star_pred.reshape(domain3d.Nt, domain3d.Nx, domain3d.Ny, domain3d.Nz)
        
    inference_time = time.time() - inf_start
    if output_dir:
        os.makedirs(os.path.join(output_dir, "PINN", "3D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "PINN", "3D", "pinn_temperature_3d.npz"), U=u_pred, x=domain3d.x, y=domain3d.y, z=domain3d.z, t=domain3d.t)
        
    return u_pred, pinn_time, inference_time, losses, model
