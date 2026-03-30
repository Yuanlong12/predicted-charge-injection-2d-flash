import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
# constants
e=1.6e-19                   # C (elementary charge)
kb=1.38e-23                 # J/K (Boltzmann constant)
T=300                       # K (temperature)
kbT=0.026                   # V (thermal voltage)
m0 = 9.11e-31               # electron rest mass (kg)
eps_ox = 3.4                # tunnel layer, dielectric constant for hBN
eps0 = 8.85e-14             # vacuum permittivity (F/cm)
e = 1.6e-19                 # electron charge (C)
h_planck = 6.626e-34        # Planck constant (J s)

def coeff(m, phi):
    A = e**3/(8*np.pi*h_planck*phi*e)
    B = -8/3*np.pi*np.sqrt(2*m*m0)*(phi*e)**(3/2)/(h_planck*e)
    return A, B

# Generate a segmented pulse waveform to facilitate numerical solving with solve_ivp.
# t_uplim is the total simulation time, t_pulse is the pulse width,
# V_pulse is applied within the pulse window, and 0 is applied outside the pulse window.
def segments_to_pulse(t_pulse, V_pulse, t_llim=1e-9, t_uplim=100):
    t_spans = []
    Max_steps = []
    V_cg = []

    # Step 1: Generate segmented time intervals from t_llim to t_pulse
    low = t_llim
    while low < t_pulse:
        up = min(low * 10, t_pulse)  # Make sure the upper bound does not exceed t_pulse
        t_spans.append((low, up))
        Max_steps.append(low / 100)
        V_cg.append(V_pulse)
        low = up

    # Step 2: Generate segmented time intervals from t_pulse to t_uplim
    while low < t_uplim:
        up = min(low * 10, t_uplim)
        t_spans.append((low, up))
        Max_steps.append(low / 100)
        V_cg.append(0)
        low = up

    return t_spans, Max_steps, V_cg

# For voltage sweep calculations, generate the time segments corresponding to each voltage step.
# For example, if the dwell time of each voltage step is 10 ms, then t_uplim = 10 ms.
def segments_to_step(t_uplim, t_llim=1e-9):
    t_spans = []
    Max_steps =[]
    low = t_llim
    for i in range (int(np.log10(t_uplim/t_llim))):
        up = low*10
        t_spans.append((low, up))
        Max_steps.append(low/100)
        low = up
    return t_spans, Max_steps

# Generate the voltage list for bidirectional sweep mode.
# The holding time of each voltage point is the above t_step = t_uplim.
def dual_sweep(init, low, high, stepsize):
    V = []
    V.extend(np.linspace(init, high, int((high-init)/stepsize), endpoint=False))
    V.extend(np.linspace(high, low, int((high-low)/stepsize), endpoint=False))
    V.extend(np.linspace(low, high, int((high-low)/stepsize), endpoint=False))
    return np.array(V)

# Define the floating-gate transistor device object
class FloatingGateTransistor:
    def __init__(self, t_ox=8.4e-9, cell_area=1e-10, GCR=0.625, m_p=1.33, phi_p=1.975, m_n=1.33, phi_n=1.84, Vth0=0, Vth_fg=0):
        self.t_ox = t_ox
        self.cell_area = cell_area
        self.GCR = GCR
        self.C_fg = eps_ox * eps0 * self.cell_area / self.t_ox  # floating-gate capacitance (F)
        self.C_cg = self.C_fg / (1 - GCR)  # control-gate capacitance (F)
        
        self.m_p = m_p
        self.phi_p = phi_p
        self.m_n = m_n
        self.phi_n = phi_n
        self.A_p, self.B_p = coeff(self.m_p, self.phi_p)  # FN tunneling current parameters under positive bias
        self.A_n, self.B_n = coeff(self.m_n, self.phi_n)  # FN tunneling current parameters under negative bias
        self.ratio = 1
        self.read_time = 100
        self.min_step_size = 1e-9  # Default minimum step size for transient solving
        
        # Initialize floating-gate charge and threshold voltage
        self.Vth_fg = Vth_fg  # Threshold calibration relative to zero floating-gate potential
        self.Vth0 = Vth0 
        self.Vth = self.Vth0  # Initial threshold voltage at zero floating-gate charge
        self.Q0 = 0  # initial zero floating-gate charge
        self.Q = -(self.Vth*self.GCR-self.Vth_fg)*(self.C_fg+self.C_cg)  # Update floating-gate charge
        self.E_ox = self.Q/(self.C_fg+self.C_cg)/self.t_ox  # initial oxide electric field
        
        self.status = (self.Vth, self.Q, self.E_ox)  # used to store the current state (Vth, Q, E_ox)
        
    def reset(self):  # Reset the device state before a new calculation
        self.Vth = self.Vth0
        self.Q = -(self.Vth*self.GCR-self.Vth_fg)*(self.C_fg+self.C_cg)
        self.Q0 = 0  # initial zero floating-gate charge
        self.E_ox = self.Q/(self.C_fg+self.C_cg)/self.t_ox
        self.status = (self.Vth, self.Q, self.E_ox)  # reset state (Vth, Q, E_ox)
        
    def dQdt(self, t, Q, Vcg):
        E_ox = (Vcg * self.GCR + Q / (self.C_cg + self.C_fg)) / self.t_ox
        if abs(E_ox) <= 1e2:
            return 0
        
        # Select the FN parameters according to the sign of E_ox
        if E_ox < 0:
            A = self.A_n
            B = self.B_n
            J = A * E_ox**2 * np.exp(B / abs(E_ox))
        else:
            A = self.A_p
            B = self.B_p
            J = A * E_ox**2 * np.exp(B / abs(E_ox))
        dQ = -np.sign(E_ox)*J * self.cell_area*self.ratio  # Determine the floating-gate charge change from the field direction
        
        return dQ

    def ProgErase_pulse(self, t_pulse, V_pulse, Vth_0=1.0, Vth_fg=0, plot = False, RESET = True): 
        # Vth_0: initial threshold voltage before the program or erase pulse
        # Vth_fg: threshold calibration under floating-gate modulation
        # ΔVth = -ΔQ / ((C_cg + C_fg) * GCR)
        if RESET:  # Reset the state if needed
            self.reset()
            self.Vth = Vth_0  # Update the initial threshold voltage
            self.Vth_fg = Vth_fg  # Update the floating-gate threshold calibration
            self.Q = -(self.Vth*self.GCR-self.Vth_fg)*(self.C_fg+self.C_cg)  # Update floating-gate charge
            
        # Store results
        t_all = []
        Q_all = []
        E_all = []

        t_span, max_step, V_cg  = segments_to_pulse(t_pulse, V_pulse, self.min_step_size, self.read_time)
        # Solve piecewise

        for t_span, max_step, V_cg in zip(t_span, max_step, V_cg):
            sol = solve_ivp(self.dQdt, t_span, [self.Q], args=(V_cg,), method='RK45', max_step=max_step)  # Use the current Q as the initial value
            t_all.extend(sol.t)
            Q_all.extend(sol.y[0])
            E_all.extend((V_cg*self.GCR+sol.y[0]/(self.C_cg+self.C_fg))/self.t_ox)
            self.Q = sol.y[0][-1]  # Record the final charge of this segment
        self.E_ox = E_all[-1]
        self.Vth = -self.Q/((self.C_cg+self.C_fg))/self.GCR+Vth_fg/self.GCR  # Update the device threshold voltage from charge
        DELTA_Vth = self.Vth - Vth_0  # Calculate the threshold voltage shift
        
        # Plot the charge and electric-field evolution under pulse operation
        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot charge evolution
            ax1.plot(t_all, Q_all, 'r-', label='injected charge')
            ax1.set_xscale('log')
            ax1.set_yscale('linear')
            ax1.set_xlabel('time (s)', fontsize=12)
            ax1.set_ylabel('Q (C)', color='r', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.grid(True, which='both', linestyle='--')

            # Create a second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            # Plot electric-field evolution
            ax2.plot(t_all, E_all, 'b--', label='electric field')
            ax2.set_ylabel('Electric field (V/m)', color='b', fontsize=12)

            ax2.tick_params(axis='y', labelcolor='b')

            # Set title
            ax1.set_title('Charge vs. Pulse Width (With Floating Gate Feedback)', fontsize=14)

            # Show legend
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

            plt.show()
        
        self.status = (self.Vth, self.Q, self.E_ox)  # Update state
        return DELTA_Vth, t_all, Q_all, E_all  # Return threshold voltage shift, time, charge, and electric field
    
    def charge_stepwise(self, V_cg, t_step):  # No relaxation is considered, suitable for hysteresis sweep
        # Store results
        t_all = []
        Q_all = []
        E_all = []
            
        # Solve piecewise
        t_spans, max_stepsizes = segments_to_step(t_step, t_llim=1e-9)

        for t_span, max_step in zip(t_spans, max_stepsizes):
            sol = solve_ivp(self.dQdt, t_span, [self.Q], args=(V_cg,), method='RK45', max_step=max_step)
            t_all.extend(sol.t)
            Q_all.extend(sol.y[0])
            E_all.extend((V_cg*self.GCR+sol.y[0]/(self.C_cg+self.C_fg))/self.t_ox)
            self.Q = sol.y[0][-1]  # Update the initial charge for the next segment
        self.E = E_all[-1]

        self.status = (self.status[0], self.Q, self.E)  # Update state without changing Vth
        return self.E*self.t_ox  # Return the corresponding potential
    
    def hysteresis_sweep(self, V_low, V_high, step_size, t_step, plot=None):
        # Generate the voltage sequence for bidirectional sweep
        V_list = dual_sweep(0, V_low, V_high, step_size)
        
        # Store results
        Q_all = []
        E_all = []
        for V in V_list:
            self.charge_stepwise(V, t_step)

            Q_all.append(self.status[1])  # Store charge
            E_all.append(self.status[2])  # Store electric field
        
        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot charge evolution
            ax1.plot(V_list, Q_all, 'r-', label='injected charge')
            ax1.set_yscale('linear')
            ax1.set_xlabel('V_cg (V)', fontsize=12)
            ax1.set_ylabel('Q (C)', color='r', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.grid(True, which='both', linestyle='--')

            # Create a second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            # Plot electric-field evolution
            ax2.plot(V_list, E_all, 'b--', label='electric field')
            ax2.set_ylabel('Electric field (V/m)', color='b', fontsize=12)

            ax2.tick_params(axis='y', labelcolor='b')

            # Set title
            ax1.set_title('Charge vs. Pulse Width (With Floating Gate Feedback)', fontsize=14)

            # Show legend
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

            plt.show()
        max_E = np.max(E_all)
        min_E = np.min(E_all)
        print(f"Max electric field: {max_E} V/m")
        print(f"Min electric field: {min_E} V/m")
        return max_E, min_E
            
    def Prog_Erase_map(self, V_progs, t_pulses, Vth_0=0, Vth_fg=0):
        # Store results
        Vth = []
        for V_prog in V_progs:
            
            for t_pulse in t_pulses:
                vth = self.ProgErase_pulse(t_pulse, V_prog, Vth_0=Vth_0, Vth_fg=Vth_fg, plot=False)
                Vth.append(vth)
        
        return np.array(Vth)
    
    def float_gate_potential(self, V_cg = 10, V_step = 2, t_step = 1e-1, tol = 0.05, iter = 5):
          
        # Positive sweep for Vfg_pos
        self.reset()
        for i in range(iter):
            Vfg_0=self.charge_stepwise(V_cg+i*V_step, t_step=t_step)
            Vfg_pos=self.charge_stepwise(V_cg+(i+1)*V_step, t_step=t_step)
            if abs((Vfg_pos-Vfg_0)/Vfg_pos) <= tol:
                break
        # Negative sweep for Vfg_neg
        self.reset()
        for i in range(iter):
            Vfg_0=self.charge_stepwise(-(V_cg+i*V_step), t_step=t_step)
            Vfg_neg=self.charge_stepwise(-(V_cg+(i+1)*V_step), t_step=t_step)
            if abs((Vfg_neg-Vfg_0)/Vfg_neg) <= tol:
                break        
        
        return Vfg_pos, Vfg_neg    

def iter_callback(params, iter, resid, *args, **kwargs):
    
    global best_params, best_residual

    current_residual = np.sum(resid**2)

    if current_residual < best_residual:
        best_residual = current_residual
        best_params = params.copy()  # Save the current best parameters

    if iter > 0 and iter % 1 == 0:
        print(f"Iteration {iter}, Residual: {current_residual:.6f}")

def _process_single_point(args):  # Helper function for single-point multi-core parallel computation
    V_prog_i, t_prog_i, Vth_ref_i, phi_p, m_p, phi_n, m_n, Vth_fg, t_ox, GCR = args
    x = FloatingGateTransistor(m_p=m_p, phi_p=phi_p, m_n=m_n, phi_n=phi_n, 
                               Vth_fg=Vth_fg,
                               t_ox=t_ox,
                               GCR=GCR)
                               
    x.read_time = 5
    Delta_Vth_i, _, _, _ = x.ProgErase_pulse(t_prog_i, V_prog_i, Vth_0=Vth_ref_i, plot=False)
    return Delta_Vth_i

if __name__ == "__main__":
    # Test code
    fgt = FloatingGateTransistor(GCR = 0.625)
    Vth = fgt.ProgErase_pulse(1e-6, -20, Vth_0=0, Vth_fg0=0, plot=False)
    print(f"Threshold voltage after programming: {Vth} V")
    
    # Test floating-gate potential
    Vfg_pos, Vfg_neg = fgt.float_gate_potential(V_cg=10, V_step=2, t_step=1e-1)
    print(f"Positive float gate potential: {Vfg_pos} V")
    print(f"Negative float gate potential: {Vfg_neg} V")
