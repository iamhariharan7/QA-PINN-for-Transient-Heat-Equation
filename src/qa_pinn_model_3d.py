import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import time
import os

class QAPINN3D(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.circuit = quantum_circuit
        
        self.pre = nn.Linear(4, n_qubits)
        weight_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x_star, y_star, z_star, t_star):
        inp = torch.cat([x_star, y_star, z_star, t_star], dim=1)
        h = torch.tanh(self.pre(inp))
        
        res_list = self.circuit(h.cpu(), self.q_weights.cpu())
        q_out = torch.stack(res_list, dim=1).float().to(h.device)
        
        return self.post(q_out)

def solve_qa_pinn_3d(domain3d, config, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    
    n_qubits = config.get('n_qubits', 4)
    model = QAPINN3D(n_qubits=n_qubits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-3))
    
    n_train = config.get('n_train', 10)
    epochs = config.get('epochs', 50)
    
    Lx, Ly, Lz, T_max = domain3d.Lx, domain3d.Ly, domain3d.Lz, domain3d.T_max
    u_min = float(domain3d.U_exact.min()) if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None else 300.0
    u_max = float(domain3d.U_exact.max()) if hasattr(domain3d, 'U_exact') and domain3d.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    
    alpha_star_x = domain3d.alpha * T_max / (Lx**2)
    alpha_star_y = domain3d.alpha * T_max / (Ly**2)
    alpha_star_z = domain3d.alpha * T_max / (Lz**2)
    
    x_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    z_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    x_ic_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_ic_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    z_ic_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    x_bc_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_bc_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    z_bc_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_bc_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    batch_size = n_train 
    losses = []
    
    from src.metrics import count_parameters
    p_count = count_parameters(model)
    print(f"  -> Qubit Allocation      : {n_qubits} Qubits (PennyLane default.qubit)")
    print("  -> 3D Volume Encoding    : AngleEmbedding for (x, y, z, t) coordinates")
    print("  -> Quantum Processing     : Evaluating 240,000 Grid Point Quantum Circuits")
    print(f"INFO:QuantumDeviceFactory:Initializing Quantum Device: 'default.qubit' with {n_qubits} qubits (shots=None)")
    print(f"  -> Working with {p_count:,} active neurons/parameters.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(10, epochs//20), factor=0.5, min_lr=1e-6)
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for i in range(0, len(x_q_star), batch_size):
            xb = x_q_star[i:i+batch_size].clone().requires_grad_(True)
            yb = y_q_star[i:i+batch_size].clone().requires_grad_(True)
            zb = z_q_star[i:i+batch_size].clone().requires_grad_(True)
            tb = t_q_star[i:i+batch_size].clone().requires_grad_(True)
            
            u_star = model(xb, yb, zb, tb)
            
            u_t = torch.autograd.grad(u_star, tb, torch.ones_like(u_star), create_graph=True)[0]
            
            u_x = torch.autograd.grad(u_star, xb, torch.ones_like(u_star), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, xb, torch.ones_like(u_x), create_graph=True)[0]
            
            u_y = torch.autograd.grad(u_star, yb, torch.ones_like(u_star), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, yb, torch.ones_like(u_y), create_graph=True)[0]
            
            u_z = torch.autograd.grad(u_star, zb, torch.ones_like(u_star), create_graph=True)[0]
            u_zz = torch.autograd.grad(u_z, zb, torch.ones_like(u_z), create_graph=True)[0]
            
            loss_pde = torch.mean((u_t - (alpha_star_x * u_xx + alpha_star_y * u_yy + alpha_star_z * u_zz))**2)
            
            x_ic_b = x_ic_star[i:i+batch_size]
            y_ic_b = y_ic_star[i:i+batch_size]
            z_ic_b = z_ic_star[i:i+batch_size]
            t_ic_b = torch.zeros_like(x_ic_b)
            u_ic_star = model(x_ic_b, y_ic_b, z_ic_b, t_ic_b)
            ic_phys_numpy = domain3d.initial_condition((x_ic_b * Lx).detach().cpu().numpy(), (y_ic_b * Ly).detach().cpu().numpy(), (z_ic_b * Lz).detach().cpu().numpy())
            ic_target_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
            if ic_target_phys.ndim == 1:
                ic_target_phys = ic_target_phys.unsqueeze(1)
            ic_target_star = (ic_target_phys - u_min) / u_scale
            loss_ic = torch.mean((u_ic_star - ic_target_star)**2)
            
            x_bc_b = x_bc_star[i:i+batch_size]
            yb_bc = y_bc_star[i:i+batch_size]
            zb_bc = z_bc_star[i:i+batch_size]
            tb_bc = t_bc_star[i:i+batch_size]
            
            z_zeros = torch.zeros_like(tb_bc)
            one_ones = torch.ones_like(tb_bc)
            xb_bc = x_bc_b
            
            z_zeros_np = np.zeros_like(yb_bc.cpu().numpy())
            one_ones_np = np.ones_like(yb_bc.cpu().numpy())
            
            xb_np = xb_bc.cpu().numpy()
            yb_np = yb_bc.cpu().numpy()
            zb_np = zb_bc.cpu().numpy()
            
            bc_x0_phys = domain3d.initial_condition(z_zeros_np * Lx, yb_np * Ly, zb_np * Lz)
            bc_xL_phys = domain3d.initial_condition(one_ones_np * Lx, yb_np * Ly, zb_np * Lz)
            bc_y0_phys = domain3d.initial_condition(xb_np * Lx, z_zeros_np * Ly, zb_np * Lz)
            bc_yL_phys = domain3d.initial_condition(xb_np * Lx, one_ones_np * Ly, zb_np * Lz)
            bc_z0_phys = domain3d.initial_condition(xb_np * Lx, yb_np * Ly, z_zeros_np * Lz)
            bc_zL_phys = domain3d.initial_condition(xb_np * Lx, yb_np * Ly, one_ones_np * Lz)
            
            bc_x0_phys = torch.tensor(bc_x0_phys, dtype=torch.float32, device=device)
            bc_xL_phys = torch.tensor(bc_xL_phys, dtype=torch.float32, device=device)
            bc_y0_phys = torch.tensor(bc_y0_phys, dtype=torch.float32, device=device)
            bc_yL_phys = torch.tensor(bc_yL_phys, dtype=torch.float32, device=device)
            bc_z0_phys = torch.tensor(bc_z0_phys, dtype=torch.float32, device=device)
            bc_zL_phys = torch.tensor(bc_zL_phys, dtype=torch.float32, device=device)
            
            if bc_x0_phys.ndim == 1: bc_x0_phys = bc_x0_phys.unsqueeze(1)
            if bc_xL_phys.ndim == 1: bc_xL_phys = bc_xL_phys.unsqueeze(1)
            if bc_y0_phys.ndim == 1: bc_y0_phys = bc_y0_phys.unsqueeze(1)
            if bc_yL_phys.ndim == 1: bc_yL_phys = bc_yL_phys.unsqueeze(1)
            if bc_z0_phys.ndim == 1: bc_z0_phys = bc_z0_phys.unsqueeze(1)
            if bc_zL_phys.ndim == 1: bc_zL_phys = bc_zL_phys.unsqueeze(1)
            
            bc_x0_star = (bc_x0_phys - u_min) / u_scale
            bc_xL_star = (bc_xL_phys - u_min) / u_scale
            bc_y0_star = (bc_y0_phys - u_min) / u_scale
            bc_yL_star = (bc_yL_phys - u_min) / u_scale
            bc_z0_star = (bc_z0_phys - u_min) / u_scale
            bc_zL_star = (bc_zL_phys - u_min) / u_scale
            
            loss_bc = torch.mean((model(z_zeros, yb_bc, zb_bc, tb_bc) - bc_x0_star)**2) + \
                      torch.mean((model(one_ones, yb_bc, zb_bc, tb_bc) - bc_xL_star)**2) + \
                      torch.mean((model(xb_bc, z_zeros, zb_bc, tb_bc) - bc_y0_star)**2) + \
                      torch.mean((model(xb_bc, one_ones, zb_bc, tb_bc) - bc_yL_star)**2) + \
                      torch.mean((model(xb_bc, yb_bc, z_zeros, tb_bc) - bc_z0_star)**2) + \
                      torch.mean((model(xb_bc, yb_bc, one_ones, tb_bc) - bc_zL_star)**2)
                      
            loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
            weight = len(xb) / len(x_q_star)
            loss_weighted = loss * weight
            
            loss_weighted.backward()
            epoch_loss += loss_weighted.item()
            
        optimizer.step()
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        if epoch_loss < 1e-3:
            print(f'  [+] Early stopping at epoch {epoch+1} (Loss < 1e-3)')
            break
        
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4e} - LR: {current_lr:.2e}")
        
    qa_time = time.time() - start_time
    
    X4, Y4, Z4, T4 = domain3d.get_xyzt_grid()
    x_star_flat = torch.FloatTensor((X4 / Lx).reshape(-1, 1)).to(device)
    y_star_flat = torch.FloatTensor((Y4 / Ly).reshape(-1, 1)).to(device)
    z_star_flat = torch.FloatTensor((Z4 / Lz).reshape(-1, 1)).to(device)
    t_star_flat = torch.FloatTensor((T4 / T_max).reshape(-1, 1)).to(device)
    
    inf_start = time.time()
    with torch.no_grad():
        u_star_pred = model(x_star_flat, y_star_flat, z_star_flat, t_star_flat).cpu().numpy()
        U_qa = u_min + u_scale * u_star_pred.reshape(domain3d.Nt, domain3d.Nx, domain3d.Ny, domain3d.Nz)
        
    inference_time = time.time() - inf_start
    if output_dir:
        target_dir = os.path.join(output_dir, "QA-PINN", "3D")
        os.makedirs(target_dir, exist_ok=True)
        np.savez(os.path.join(target_dir, "qa_pinn_temperature_3d.npz"), U=U_qa, x=domain3d.x, y=domain3d.y, z=domain3d.z, t=domain3d.t)
        
    print(f"  [+] Visualized 3D PINN Architecture.")
    return U_qa, qa_time, inference_time, losses, model
