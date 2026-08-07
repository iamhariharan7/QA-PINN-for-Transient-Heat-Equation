import torch
import torch.nn as nn
import numpy as np
import time
import os

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x_star, t_star):
        inp = torch.cat([x_star, t_star], dim=1)
        return self.net(inp)

def solve_pinn(domain, config, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    
    n_train = config.get('n_train', 500)
    epochs = config.get('epochs', 500)
    
    L = domain.L
    T_max = domain.T_max
    u_min = float(domain.U_exact.min()) if hasattr(domain, 'U_exact') and domain.U_exact is not None else 300.0
    u_max = float(domain.U_exact.max()) if hasattr(domain, 'U_exact') and domain.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    alpha_star = domain.alpha * T_max / (L**2)
    
    # Non-dimensionalized coordinates [0, 1]
    x_phys_star = torch.FloatTensor(n_train, 1).uniform_(0, 1)
    t_phys_star = torch.FloatTensor(n_train, 1).uniform_(0, 1)
    x_ic_star = torch.FloatTensor(n_train, 1).uniform_(0, 1)
    t_bc_star = torch.FloatTensor(n_train, 1).uniform_(0, 1)
    

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
        x_phys_star.requires_grad_(True)
        t_phys_star.requires_grad_(True)
        u_star = model(x_phys_star, t_phys_star)
        
        u_t_star = torch.autograd.grad(u_star, t_phys_star, torch.ones_like(u_star), create_graph=True)[0]
        u_x_star = torch.autograd.grad(u_star, x_phys_star, torch.ones_like(u_star), create_graph=True)[0]
        u_xx_star = torch.autograd.grad(u_x_star, x_phys_star, torch.ones_like(u_x_star), create_graph=True)[0]
        
        loss_pde = torch.mean((u_t_star - alpha_star * u_xx_star)**2)
        
        t_ic_star = torch.zeros_like(x_ic_star)
        u_ic_star = model(x_ic_star, t_ic_star)
        ic_phys_numpy = domain.initial_condition((x_ic_star * L).detach().cpu().numpy())
        target_ic_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
        if target_ic_phys.ndim == 1:
            target_ic_phys = target_ic_phys.unsqueeze(1)
        target_ic_star = (target_ic_phys - u_min) / u_scale
        loss_ic = torch.mean((u_ic_star - target_ic_star)**2)
        
        x0_star = torch.zeros_like(t_bc_star)
        x1_star = torch.ones_like(t_bc_star)
        left_bc_val_numpy = domain.initial_condition(np.zeros((1, 1)))[0]
        right_bc_val_numpy = domain.initial_condition(np.full((1, 1), L))[0]
        left_bc_val = torch.tensor((left_bc_val_numpy - u_min) / u_scale, dtype=torch.float32, device=device)
        right_bc_val = torch.tensor((right_bc_val_numpy - u_min) / u_scale, dtype=torch.float32, device=device)
        loss_bc = torch.mean((model(x0_star, t_bc_star) - left_bc_val)**2) + torch.mean((model(x1_star, t_bc_star) - right_bc_val)**2)
        
        return loss_pde + 100.0 * loss_ic + 100.0 * loss_bc

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(10, epochs//20), factor=0.5, min_lr=1e-6)
    
    losses = []
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
    
    X, T, x, t = domain.get_grid()
    X_star = torch.FloatTensor(X.flatten() / L).unsqueeze(1).to(device)
    T_star = torch.FloatTensor(T.flatten() / T_max).unsqueeze(1).to(device)
    
    inf_start = time.time()
    with torch.no_grad():
        U_star_pred = model(X_star, T_star).cpu().numpy().reshape(domain.Nt, domain.Nx)
        U_pinn = u_min + u_scale * U_star_pred
        
    inference_time = time.time() - inf_start
    if output_dir:
        os.makedirs(os.path.join(output_dir, "PINN", "1D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "PINN", "1D", "pinn_temperature.npz"), U=U_pinn, x=x, t=t)
        
    return U_pinn, pinn_time, inference_time, losses, model
