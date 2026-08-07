import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import time
import os

class QAPINN2D(nn.Module):
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
        
        self.pre = nn.Linear(3, n_qubits)
        weight_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x_star, y_star, t_star):
        inp = torch.cat([x_star, y_star, t_star], dim=1)
        h = torch.tanh(self.pre(inp))
        
        res_list = self.circuit(h.cpu(), self.q_weights.cpu())
        q_out = torch.stack(res_list, dim=1).float().to(h.device)
        
        return self.post(q_out)

def solve_qa_pinn_2d(domain2d, config, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    
    n_qubits = config.get('n_qubits', 4)
    model = QAPINN2D(n_qubits=n_qubits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-3))
    
    n_train = config.get('n_train', 20)
    epochs = config.get('epochs', 100)
    
    Lx, Ly, T_max = domain2d.Lx, domain2d.Ly, domain2d.T_max
    u_min = float(domain2d.U_exact.min()) if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None else 300.0
    u_max = float(domain2d.U_exact.max()) if hasattr(domain2d, 'U_exact') and domain2d.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    
    alpha_star_x = domain2d.alpha * T_max / (Lx**2)
    alpha_star_y = domain2d.alpha * T_max / (Ly**2)
    
    x_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    x_bc_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_bc_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_bc_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    x_ic_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    y_ic_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
    batch_size = n_train
    losses = []
    
    from src.metrics import count_parameters
    p_count = count_parameters(model)
    print(f"  -> Qubit Allocation      : {n_qubits} Qubits (PennyLane default.qubit)")
    print("  -> Quantum State Encoding : AngleEmbedding (Hadamard + Rx Gate Mapping)")
    print("  -> Variational Ansatz     : BasicEntanglerLayers (2 Layers of Ry/Rz Rotations + CNOT Rings)")
    print(f"  -> Quantum Observables    : PauliZ Expectation Values <Z_i> for i in [0..{n_qubits-1}]")
    print(f"INFO:QuantumDeviceFactory:Initializing Quantum Device: 'default.qubit' with {n_qubits} qubits (shots=None)")
    print(f"  -> Working with {p_count:,} active neurons/parameters.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(10, epochs//20), factor=0.5, min_lr=1e-6)
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for i in range(0, len(x_q_star), batch_size):
            xb = x_q_star[i:i+batch_size].clone().requires_grad_(True)
            yb = y_q_star[i:i+batch_size].clone().requires_grad_(True)
            tb = t_q_star[i:i+batch_size].clone().requires_grad_(True)
            
            u_star = model(xb, yb, tb)
            u_t = torch.autograd.grad(u_star, tb, torch.ones_like(u_star), create_graph=True)[0]
            
            u_x = torch.autograd.grad(u_star, xb, torch.ones_like(u_star), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, xb, torch.ones_like(u_x), create_graph=True)[0]
            
            u_y = torch.autograd.grad(u_star, yb, torch.ones_like(u_star), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, yb, torch.ones_like(u_y), create_graph=True)[0]
            
            loss_pde = torch.mean((u_t - (alpha_star_x * u_xx + alpha_star_y * u_yy))**2)
            
            x_ic_b = x_ic_q_star[i:i+batch_size]
            y_ic_b = y_ic_q_star[i:i+batch_size]
            t_ic_b = torch.zeros_like(x_ic_b)
            u_ic_star = model(x_ic_b, y_ic_b, t_ic_b)
            ic_phys_numpy = domain2d.initial_condition((x_ic_b * Lx).detach().cpu().numpy(), (y_ic_b * Ly).detach().cpu().numpy())
            ic_target_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
            if ic_target_phys.ndim == 1:
                ic_target_phys = ic_target_phys.unsqueeze(1)
            ic_target_star = (ic_target_phys - u_min) / u_scale
            loss_ic = torch.mean((u_ic_star - ic_target_star)**2)
            
            y_bc_b = y_bc_q_star[i:i+batch_size]
            x_bc_b = x_bc_q_star[i:i+batch_size]
            t_bc_b = t_bc_q_star[i:i+batch_size]
            z_zeros = torch.zeros_like(t_bc_b)
            one_ones = torch.ones_like(t_bc_b)
            
            bc0_x_phys = domain2d.initial_condition(np.zeros_like(y_bc_b.cpu().numpy()) * Lx, (y_bc_b * Ly).detach().cpu().numpy())
            bcL_x_phys = domain2d.initial_condition(np.ones_like(y_bc_b.cpu().numpy()) * Lx, (y_bc_b * Ly).detach().cpu().numpy())
            bc0_y_phys = domain2d.initial_condition((x_bc_b * Lx).detach().cpu().numpy(), np.zeros_like(x_bc_b.cpu().numpy()) * Ly)
            bcL_y_phys = domain2d.initial_condition((x_bc_b * Lx).detach().cpu().numpy(), np.ones_like(x_bc_b.cpu().numpy()) * Ly)
            
            bc0_x_phys = torch.tensor(bc0_x_phys, dtype=torch.float32, device=device)
            bcL_x_phys = torch.tensor(bcL_x_phys, dtype=torch.float32, device=device)
            bc0_y_phys = torch.tensor(bc0_y_phys, dtype=torch.float32, device=device)
            bcL_y_phys = torch.tensor(bcL_y_phys, dtype=torch.float32, device=device)
            
            if bc0_x_phys.ndim == 1: bc0_x_phys = bc0_x_phys.unsqueeze(1)
            if bcL_x_phys.ndim == 1: bcL_x_phys = bcL_x_phys.unsqueeze(1)
            if bc0_y_phys.ndim == 1: bc0_y_phys = bc0_y_phys.unsqueeze(1)
            if bcL_y_phys.ndim == 1: bcL_y_phys = bcL_y_phys.unsqueeze(1)
            
            bc0_x = (bc0_x_phys - u_min) / u_scale
            bcL_x = (bcL_x_phys - u_min) / u_scale
            bc0_y = (bc0_y_phys - u_min) / u_scale
            bcL_y = (bcL_y_phys - u_min) / u_scale
            
            loss_bc = torch.mean((model(z_zeros, y_bc_b, t_bc_b) - bc0_x)**2) + \
                      torch.mean((model(one_ones, y_bc_b, t_bc_b) - bcL_x)**2) + \
                      torch.mean((model(x_bc_b, z_zeros, t_bc_b) - bc0_y)**2) + \
                      torch.mean((model(x_bc_b, one_ones, t_bc_b) - bcL_y)**2)
                      
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
    
    X3, Y3, T3 = domain2d.get_xyt_grid()
    x_star_flat = torch.FloatTensor((X3 / Lx).reshape(-1, 1)).to(device)
    y_star_flat = torch.FloatTensor((Y3 / Ly).reshape(-1, 1)).to(device)
    t_star_flat = torch.FloatTensor((T3 / T_max).reshape(-1, 1)).to(device)
    
    inf_start = time.time()
    with torch.no_grad():
        u_star_pred = model(x_star_flat, y_star_flat, t_star_flat).cpu().numpy()
        U_qa = u_min + u_scale * u_star_pred.reshape(domain2d.Nt, domain2d.Ny, domain2d.Nx)
        
    inference_time = time.time() - inf_start
    if output_dir:
        os.makedirs(os.path.join(output_dir, "QA-PINN", "2D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "QA-PINN", "2D", "qa_pinn_temperature_2d.npz"), U=U_qa, x=domain2d.x, y=domain2d.y, t=domain2d.t)
        
    print("  [+] Visualized 2D PINN Architecture.")
    return U_qa, qa_time, inference_time, losses, model
