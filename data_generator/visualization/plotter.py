import os
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    plt = None

class Plotter:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def plot_1d(self, sim_id: str, split_dir: str, t: np.ndarray, x: np.ndarray, T: np.ndarray):
        if plt is None: return
        out_dir = os.path.join(self.run_dir, split_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        # Plot initial, middle, and final states
        indices = [0, len(t)//2, -1]
        for idx in indices:
            plt.plot(x, T[idx], label=f"t={t[idx]:.2f}s")
        plt.xlabel("Position (m)")
        plt.ylabel("Temperature (K)")
        plt.title(f"1D Heat Conduction (Sim: {sim_id})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{sim_id}_curves.png"), dpi=150)
        plt.close()

    def plot_2d_heatmap(self, sim_id: str, split_dir: str, t: np.ndarray, x: np.ndarray, y: np.ndarray, T: np.ndarray):
        if plt is None: return
        out_dir = os.path.join(self.run_dir, split_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        # Final frame
        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(T[-1], extent=extent, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(label='Temperature (K)')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"2D Heatmap at t={t[-1]:.2f}s (Sim: {sim_id})")
        plt.savefig(os.path.join(out_dir, f"{sim_id}_heatmap.png"), dpi=150)
        plt.close()

    def create_1d_animation(self, sim_id: str, split_dir: str, t: np.ndarray, x: np.ndarray, T: np.ndarray):
        if plt is None: return
        out_dir = os.path.join(self.run_dir, split_dir, "animations")
        os.makedirs(out_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(x, T[0])
        ax.set_ylim(T.min() - 5, T.max() + 5)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Temperature (K)")
        title = ax.set_title(f"t={t[0]:.2f}s")
        
        def update(frame):
            line.set_ydata(T[frame])
            title.set_text(f"t={t[frame]:.2f}s")
            return line, title
            
        anim = FuncAnimation(fig, update, frames=len(t), blit=True)
        try:
            anim.save(os.path.join(out_dir, f"{sim_id}_anim.gif"), writer='pillow', fps=10)
        except:
            pass
        plt.close()
