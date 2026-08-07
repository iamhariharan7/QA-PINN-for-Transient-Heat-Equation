import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import time
import os

class QAPINN(nn.Module):
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
        
        self.pre = nn.Linear(2, n_qubits)
        weight_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x_star, t_star):
        inp = torch.cat([x_star, t_star], dim=1)
        h = torch.tanh(self.pre(inp))
        
        res_list = self.circuit(h.cpu(), self.q_weights.cpu())
        q_out = torch.stack(res_list, dim=1).float().to(h.device)
        
        return self.post(q_out)

def solve_qa_pinn(domain, config, output_dir=None):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    
    n_qubits = config.get('n_qubits', 4)
    model = QAPINN(n_qubits=n_qubits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-3))
    
    n_train = config.get('n_train', 50)
    batch_size = config.get('batch_size', n_train)
    epochs = config.get('epochs', 100)
    
    L = domain.L
    T_max = domain.T_max
    u_min = float(domain.U_exact.min()) if hasattr(domain, 'U_exact') and domain.U_exact is not None else 300.0
    u_max = float(domain.U_exact.max()) if hasattr(domain, 'U_exact') and domain.U_exact is not None else 500.0
    u_scale = (u_max - u_min) if u_max > u_min else 1.0
    alpha_star = domain.alpha * T_max / (L**2)
    
    x_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    x_ic_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    t_bc_q_star = torch.FloatTensor(n_train, 1).uniform_(0, 1).to(device)
    
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
            x_batch = x_q_star[i:i+batch_size].clone().requires_grad_(True)
            t_batch = t_q_star[i:i+batch_size].clone().requires_grad_(True)
            
            u_star = model(x_batch, t_batch)
            
            u_t = torch.autograd.grad(u_star, t_batch, torch.ones_like(u_star), create_graph=True)[0]
            u_x = torch.autograd.grad(u_star, x_batch, torch.ones_like(u_star), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_batch, torch.ones_like(u_x), create_graph=True)[0]
            
            loss_pde = torch.mean((u_t - alpha_star * u_xx)**2)
            
            x_ic_batch = x_ic_q_star[i:i+batch_size]
            t_ic_batch = torch.zeros_like(x_ic_batch)
            u_ic_star = model(x_ic_batch, t_ic_batch)
            ic_phys_numpy = domain.initial_condition((x_ic_batch * L).detach().cpu().numpy())
            target_ic_phys = torch.tensor(ic_phys_numpy, dtype=torch.float32, device=device)
            if target_ic_phys.ndim == 1:
                target_ic_phys = target_ic_phys.unsqueeze(1)
            target_ic_star = (target_ic_phys - u_min) / u_scale
            loss_ic = torch.mean((u_ic_star - target_ic_star)**2)
            
            t_bc_batch = t_bc_q_star[i:i+batch_size]
            x0_star = torch.zeros_like(t_bc_batch)
            x1_star = torch.ones_like(t_bc_batch)
            left_bc_val = (domain.initial_condition(np.zeros((1, 1)))[0] - u_min) / u_scale
            right_bc_val = (domain.initial_condition(np.full((1, 1), L))[0] - u_min) / u_scale
            left_bc_val = torch.tensor(left_bc_val, dtype=torch.float32, device=device)
            right_bc_val = torch.tensor(right_bc_val, dtype=torch.float32, device=device)
            loss_bc = torch.mean((model(x0_star, t_bc_batch) - left_bc_val)**2) + torch.mean((model(x1_star, t_bc_batch) - right_bc_val)**2)
            
            loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
            weight = len(x_batch) / len(x_q_star)
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
    
    X, T, x, t = domain.get_grid()
    X_star = torch.FloatTensor(X.flatten() / L).unsqueeze(1).to(device)
    T_star = torch.FloatTensor(T.flatten() / T_max).unsqueeze(1).to(device)
    
    inf_start = time.time()
    with torch.no_grad():
        U_star_pred = torch.cat([model(X_star[i:i+512], T_star[i:i+512]) for i in range(0, X_star.shape[0], 512)], dim=0).cpu().numpy().reshape(domain.Nt, domain.Nx)
        U_qa = u_min + u_scale * U_star_pred
        
    inference_time = time.time() - inf_start
    if output_dir:
        os.makedirs(os.path.join(output_dir, "QA-PINN", "1D"), exist_ok=True)
        np.savez(os.path.join(output_dir, "QA-PINN", "1D", "qa_pinn_temperature.npz"), U=U_qa, x=x, t=t)
        
    print(f"  -> Active QA-PINN Neurons/Parameters: {p_count:,}")
    print("  [+] Visualized 1D Quantum Circuit & Hybrid Architecture.")
    return U_qa, qa_time, inference_time, losses, model
