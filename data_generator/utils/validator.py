import numpy as np

class Validator:
    @staticmethod
    def check_energy_conservation(initial_temp, final_temp, heat_source, dt, dx, dy=1, dz=1, rho=1, cp=1):
        """
        A simplified check for energy conservation.
        Total energy change should roughly equal the energy added by the source minus energy lost at boundaries.
        In this basic version, we just ensure temperatures don't blow up to infinity or drop below 0 K.
        """
        # Basic physical bounds check
        if np.any(final_temp < 0):
            return False, "Temperature dropped below absolute zero."
        
        if np.any(np.isnan(final_temp)) or np.any(np.isinf(final_temp)):
            return False, "Temperature contains NaN or Inf values (solver instability)."
            
        return True, "Simulation passed basic physical validation."

    @staticmethod
    def validate_cfl(dt, dx, alpha, dy=None, dz=None):
        """
        Validates that the time step satisfies the explicit von Neumann stability criterion.
        """
        criterion = alpha * dt / (dx**2)
        if dy is not None:
            criterion += alpha * dt / (dy**2)
        if dz is not None:
            criterion += alpha * dt / (dz**2)
            
        if criterion > 0.5:
            return False, f"Stability criterion violated: {criterion} > 0.5"
            
        return True, "Stability criterion satisfied."
