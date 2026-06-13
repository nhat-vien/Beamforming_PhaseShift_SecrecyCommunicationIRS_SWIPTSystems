#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO Algorithm GUI with Power Sweep and Embedded Plotting
=======================================================
Features:
- Power sweep: [18, 23, 28, 33, 38, 43] dBm
- Baseline algorithms: AO, PSO, ZF, MRT (selectable)
- Real-time embedded plotting
- Progress tracking

Author: Vien Nguyen-Duy-Nhat
Date: 2025-06-12
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import threading
import time
from datetime import datetime

# Matplotlib for embedded plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from scipy.optimize import minimize as scipy_minimize

# ============================================================================
# Channel Generation Functions (ULA-based from baseline_comparison)
# ============================================================================

def ULA(angle, M):
    """Uniform Linear Array steering vector"""
    angle_rad = np.radians(angle)
    steering_vec = np.exp(1j * np.pi * np.arange(M)[:, np.newaxis] * np.sin(angle_rad))
    return steering_vec

def cal_angle(BS_loc, RIS_loc, IU_loc, EU_loc):
    """Calculate angles between nodes"""
    def cal_angle_(A_loc, B_loc):
        rt_relative_loc = (A_loc - B_loc).T
        angle_AB = np.arctan2(rt_relative_loc[1], rt_relative_loc[0]) * 180 / np.pi
        return angle_AB

    angle_BR = cal_angle_(BS_loc, RIS_loc)
    angle_RB = cal_angle_(RIS_loc, BS_loc)
    angle_RI = cal_angle_(RIS_loc, IU_loc)
    angle_RE = cal_angle_(RIS_loc, EU_loc)
    angle_BI = cal_angle_(BS_loc, IU_loc)
    angle_BE = cal_angle_(BS_loc, EU_loc)

    return angle_BR, angle_RB, angle_RI, angle_RE, angle_BI, angle_BE

def cal_dist_total(BS_loc, RIS_loc, IU_loc, EU_loc):
    """Calculate distances between nodes"""
    def cal_dist(A_loc, B_loc):
        return np.linalg.norm(A_loc - B_loc, axis=1)

    dist_br = np.linalg.norm(BS_loc - RIS_loc)
    dist_ri = cal_dist(RIS_loc, IU_loc)
    dist_re = cal_dist(RIS_loc, EU_loc)
    dist_bi = cal_dist(BS_loc, IU_loc)
    dist_be = cal_dist(BS_loc, EU_loc)

    return dist_br, dist_ri, dist_re, dist_bi, dist_be

def generate_IU_loc(center, radius, K):
    """Generate random user locations in a circle"""
    angles = np.random.uniform(0, 2 * np.pi, K)
    radii = radius * np.sqrt(np.random.uniform(0, 1, K))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

def cal_pl(dist_br, dist_ri, dist_re, dist_bi, dist_be):
    """Calculate path loss"""
    def LOS_PL(d):
        return 30 + 20 * np.log10(d)

    h_pl_i = np.sqrt(10 ** (-LOS_PL(dist_ri) / 10))
    h_pl_e = np.sqrt(10 ** (-LOS_PL(dist_re) / 10))
    G_pl = np.sqrt(10 ** (-LOS_PL(dist_br) / 10))
    g_pl_i = np.sqrt(10 ** (-LOS_PL(dist_bi) / 10))
    g_pl_e = np.sqrt(10 ** (-LOS_PL(dist_be) / 10))

    return h_pl_i, G_pl, h_pl_e, g_pl_i, g_pl_e

def generate_channel(AP_antens, RIS_elems, angle_BR, angle_RB, angle_RI, angle_RE, angle_BI, angle_BE,
                     h_pl_i, G_pl, h_pl_e, g_pl_i, g_pl_e):
    """Generate Rician fading channels with ULA steering vectors"""
    def cal_channel(angle, Ntx, Nrx, pathloss):
        kappa = 10  # Rician K-factor
        channel_LOS = np.dot(ULA(angle, Ntx), ULA(angle, Nrx).T)
        channel_NLOS = np.sqrt(1/2) * (np.random.randn(Ntx, Nrx) + 1j * np.random.randn(Ntx, Nrx))
        channel = np.sqrt(kappa/(kappa+1)) * pathloss * channel_LOS + np.sqrt(1/(kappa+1)) * pathloss * channel_NLOS
        return channel

    hri = cal_channel(angle_RI, RIS_elems, 1, h_pl_i)
    hre = cal_channel(angle_RE, RIS_elems, 1, h_pl_e)
    gbi = cal_channel(angle_BI, 1, AP_antens, g_pl_i)
    gbt = cal_channel(angle_BE, 1, AP_antens, g_pl_e)
    G = cal_channel(angle_BR, RIS_elems, AP_antens, G_pl)

    return hri, hre, gbi, gbt, G

def generate_single_user_channels(BS_loc, RIS_loc, IU_center, EU_center, radius, K_RICIAN, M, N):
    """Generate channels for single-user scenario"""
    IU_loc = generate_IU_loc(IU_center, radius, K=1)[0]
    EU_loc = generate_IU_loc(EU_center, radius, K=1)[0]

    IU_loc_2d = IU_loc.reshape(1, -1) if IU_loc.ndim == 1 else IU_loc
    EU_loc_2d = EU_loc.reshape(1, -1) if EU_loc.ndim == 1 else EU_loc

    dist_br, dist_ri, dist_re, dist_bi, dist_be = cal_dist_total(BS_loc, RIS_loc, IU_loc_2d, EU_loc_2d)
    angle_BR, angle_RB, angle_RI, angle_RE, angle_BI, angle_BE = cal_angle(BS_loc, RIS_loc, IU_loc_2d, EU_loc_2d)

    dist_ri_val = dist_ri[0] if hasattr(dist_ri, '__len__') else dist_ri
    dist_re_val = dist_re[0] if hasattr(dist_re, '__len__') else dist_re
    dist_bi_val = dist_bi[0] if hasattr(dist_bi, '__len__') else dist_bi
    dist_be_val = dist_be[0] if hasattr(dist_be, '__len__') else dist_be

    angle_RI_val = angle_RI[0] if hasattr(angle_RI, '__len__') else angle_RI
    angle_RE_val = angle_RE[0] if hasattr(angle_RE, '__len__') else angle_RE
    angle_BI_val = angle_BI[0] if hasattr(angle_BI, '__len__') else angle_BI
    angle_BE_val = angle_BE[0] if hasattr(angle_BE, '__len__') else angle_BE

    h_pl_i, G_pl, h_pl_e, g_pl_i, g_pl_e = cal_pl(dist_br, dist_ri_val, dist_re_val, dist_bi_val, dist_be_val)

    hri_, hre_, ge_, gt_, G_ = generate_channel(M, N, angle_BR, angle_RB, angle_RI_val, angle_RE_val,
                                                 angle_BI_val, angle_BE_val, h_pl_i, G_pl, h_pl_e, g_pl_i, g_pl_e)

    return {
        'G': G_,
        'h_I': hri_.flatten(),
        'h_E': hre_.flatten(),
        'g_I': ge_.flatten(),
        'g_E': gt_.flatten()
    }

# ============================================================================
# Baseline Algorithms Class
# ============================================================================

class BaselineAlgorithms:
    """Baseline algorithms for SWIPT-IRS"""

    def __init__(self, M, N, rho, eta, sigma2):
        self.M = M
        self.N = N
        self.rho = rho
        self.eta = eta
        self.sigma2 = sigma2

    def calc_effective_channel(self, h, g, G, Phi):
        """Calculate effective channel: H_tilde = h^H * Phi * G + g^H"""
        return h.conj().T @ Phi @ G + g.conj().T

    def calc_metrics(self, w, H_I, H_E):
        """Calculate performance metrics"""
        w = w.reshape(-1, 1)

        power_I = np.abs(H_I @ w)**2
        power_E = np.abs(H_E @ w)**2

        R_I = self.rho * np.log2(1 + power_I.item() / self.sigma2)
        R_E = self.rho * np.log2(1 + power_E.item() / self.sigma2)
        R_s = max(R_I - R_E, 0.0)

        E_I_watt = (1 - self.rho) * self.eta * power_I.item()
        E_I_dBm = 10 * np.log10(E_I_watt * 1000) if E_I_watt > 1e-15 else -100

        return {'R_s': R_s, 'R_I': R_I, 'R_E': R_E, 'E_I': E_I_dBm}

    def alternating_optimization(self, channels, P_max, max_iter=20, num_trials=10, grad_iter=200, callback=None):
        """Alternating optimization"""
        global_best_metrics = {'R_s': -1}

        for trial in range(num_trials):
            if callback:
                callback(f"AO Trial {trial+1}/{num_trials}")

            try:
                theta = np.random.uniform(0, 2*np.pi, self.N)
                Phi = np.diag(np.exp(1j * theta))
                best_metrics = {'R_s': -1}

                for iteration in range(max_iter):
                    # Optimize beamforming
                    H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi)
                    H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi)

                    if CVXPY_AVAILABLE:
                        try:
                            w_var = cp.Variable(self.M, complex=True)
                            H_I_flat = H_I.flatten()
                            H_E_flat = H_E.flatten()
                            power_I = cp.real(cp.conj(H_I_flat) @ w_var)**2 + cp.imag(cp.conj(H_I_flat) @ w_var)**2
                            power_E = cp.real(cp.conj(H_E_flat) @ w_var)**2 + cp.imag(cp.conj(H_E_flat) @ w_var)**2
                            objective = cp.Maximize(power_I - power_E)
                            constraints = [cp.norm(w_var, 2)**2 <= P_max]
                            prob = cp.Problem(objective, constraints)
                            prob.solve(solver=cp.SCS, verbose=False)

                            if w_var.value is not None:
                                w = w_var.value
                                current_power = np.linalg.norm(w)**2
                                if current_power > 1e-12:
                                    w = w * np.sqrt(P_max / current_power)
                            else:
                                raise Exception("CVX failed")
                        except:
                            w = H_I.conj().T.flatten()
                            w = w * np.sqrt(P_max) / np.linalg.norm(w)
                    else:
                        w = H_I.conj().T.flatten()
                        w = w * np.sqrt(P_max) / np.linalg.norm(w)

                    # Optimize phase shifts
                    def phase_objective(theta_vec):
                        phi_vec = np.exp(1j * theta_vec)
                        Phi_temp = np.diag(phi_vec)
                        H_I_temp = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi_temp)
                        H_E_temp = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi_temp)
                        power_I = np.abs(H_I_temp @ w.reshape(-1, 1))**2
                        power_E = np.abs(H_E_temp @ w.reshape(-1, 1))**2
                        R_I = self.rho * np.log2(1 + power_I.item() / self.sigma2)
                        R_E = self.rho * np.log2(1 + power_E.item() / self.sigma2)
                        return -(R_I - R_E)

                    try:
                        result = scipy_minimize(phase_objective, theta, method='L-BFGS-B',
                                              bounds=[(0, 2*np.pi)] * self.N, options={'maxiter': grad_iter})
                        theta = result.x
                        Phi = np.diag(np.exp(1j * theta))
                    except:
                        pass

                    H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi)
                    H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi)
                    metrics = self.calc_metrics(w, H_I, H_E)

                    if metrics['R_s'] > best_metrics['R_s']:
                        best_metrics = metrics

                if best_metrics['R_s'] > global_best_metrics['R_s']:
                    global_best_metrics = best_metrics
            except:
                continue

        return global_best_metrics if global_best_metrics['R_s'] > -1 else {'R_s': 0, 'R_I': 0, 'R_E': 0, 'E_I': -100}

    def mrt_baseline(self, channels, P_max, callback=None):
        """MRT baseline"""
        if callback:
            callback("MRT")

        h_I_vec = channels['h_I'].flatten()
        G_vec = channels['G']
        theta = np.zeros(self.N)
        for n in range(self.N):
            phase_h = np.angle(h_I_vec[n])
            phase_G = np.angle(np.sum(G_vec[n, :]))
            theta[n] = -phase_h - phase_G
        Phi = np.diag(np.exp(1j * theta))

        H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi)
        w = H_I.conj().T.flatten()
        w = w * np.sqrt(P_max) / np.linalg.norm(w)

        H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi)
        return self.calc_metrics(w, H_I, H_E)

    def zero_forcing_baseline(self, channels, P_max, num_trials=5, callback=None):
        """ZF baseline"""
        if callback:
            callback("ZF")

        best_metrics = {'R_s': -1000}
        for trial in range(num_trials):
            theta = np.random.uniform(0, 2*np.pi, self.N)
            Phi = np.diag(np.exp(1j * theta))
            H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi)
            H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi)

            try:
                H_E_H = H_E.conj().T
                P_null = np.eye(self.M) - H_E_H @ np.linalg.pinv(H_E_H)
                w = (P_null @ H_I.conj().T).flatten()
                if np.linalg.norm(w) > 1e-10:
                    w = w * np.sqrt(P_max) / np.linalg.norm(w)
                else:
                    w = H_I.conj().T.flatten()
                    w = w * np.sqrt(P_max) / np.linalg.norm(w)
            except:
                w = H_I.conj().T.flatten()
                w = w * np.sqrt(P_max) / np.linalg.norm(w)

            metrics = self.calc_metrics(w, H_I, H_E)
            if metrics['R_s'] > best_metrics['R_s']:
                best_metrics = metrics

        return best_metrics

    def pso_baseline(self, channels, P_max, n_particles=20, max_iter=20, callback=None):
        """PSO baseline"""
        if callback:
            callback("PSO")

        dim_theta = self.N
        dim_w = 2 * self.M
        total_dim = dim_theta + dim_w

        # MRT initialization
        h_I_vec = channels['h_I'].flatten()
        G_vec = channels['G']
        theta_mrt = np.zeros(self.N)
        for n in range(self.N):
            phase_h = np.angle(h_I_vec[n])
            phase_G = np.angle(np.sum(G_vec[n, :]))
            theta_mrt[n] = -phase_h - phase_G
        Phi_mrt = np.diag(np.exp(1j * theta_mrt))
        H_I_mrt = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi_mrt)
        w_mrt = H_I_mrt.conj().T.flatten()
        w_mrt = w_mrt * np.sqrt(P_max) / np.linalg.norm(w_mrt)

        mrt_particle = np.zeros(total_dim)
        mrt_particle[:dim_theta] = theta_mrt
        mrt_particle[dim_theta:dim_theta + self.M] = w_mrt.real
        mrt_particle[dim_theta + self.M:] = w_mrt.imag

        # Initialize particles
        particles = np.zeros((n_particles, total_dim))
        particles[0] = mrt_particle.copy()
        for i in range(1, n_particles):
            if i < int(0.3 * n_particles):
                particles[i] = mrt_particle + np.random.normal(0, 0.1, total_dim)
                particles[i, :dim_theta] = np.clip(particles[i, :dim_theta], -np.pi, np.pi)
            else:
                particles[i] = np.random.uniform(-np.pi, np.pi, total_dim)

        velocities = np.random.uniform(-0.1, 0.1, (n_particles, total_dim))
        p_best = particles.copy()
        p_best_fitness = np.full(n_particles, -np.inf)
        g_best = particles[0].copy()
        g_best_fitness = -np.inf
        best_w = None
        best_Phi = None

        def evaluate_particle(particle):
            try:
                theta = particle[:dim_theta]
                Phi = np.diag(np.exp(1j * theta))
                w_real = particle[dim_theta:dim_theta + self.M]
                w_imag = particle[dim_theta + self.M:]
                w = (w_real + 1j * w_imag).reshape(-1, 1)
                w = w * np.sqrt(P_max) / np.linalg.norm(w)

                H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], Phi)
                H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], Phi)

                power_I = np.abs(H_I @ w) ** 2
                power_E = np.abs(H_E @ w) ** 2
                R_I = self.rho * np.log2(1 + power_I.item() / self.sigma2)
                R_E = self.rho * np.log2(1 + power_E.item() / self.sigma2)
                R_s = max(R_I - R_E, 0)

                return R_s, w, Phi
            except:
                return -np.inf, None, None

        # PSO loop
        w_inertia = 0.9
        c1 = 2.0
        c2 = 2.0

        for iteration in range(max_iter):
            for i in range(n_particles):
                fitness, w, Phi = evaluate_particle(particles[i])

                if fitness > p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best[i] = particles[i].copy()

                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best = particles[i].copy()
                    best_w = w
                    best_Phi = Phi

            for i in range(n_particles):
                r1 = np.random.rand(total_dim)
                r2 = np.random.rand(total_dim)
                velocities[i] = (w_inertia * velocities[i] +
                               c1 * r1 * (p_best[i] - particles[i]) +
                               c2 * r2 * (g_best - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i, :dim_theta] = np.clip(particles[i, :dim_theta], -np.pi, np.pi)

        if best_w is not None and best_Phi is not None:
            H_I = self.calc_effective_channel(channels['h_I'], channels['g_I'], channels['G'], best_Phi)
            H_E = self.calc_effective_channel(channels['h_E'], channels['g_E'], channels['G'], best_Phi)
            return self.calc_metrics(best_w, H_I, H_E)
        else:
            return self.mrt_baseline(channels, P_max)

# ============================================================================
# GUI Application with Embedded Plot
# ============================================================================

class PowerSweepGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Power Sweep - Baseline Comparison")
        self.root.geometry("1400x800")

        # Variables
        self.running = False
        self.results = {
            'AO': {'R_s': [], 'E_I': [], 'R_I': [], 'R_E': []},
            'PSO': {'R_s': [], 'E_I': [], 'R_I': [], 'R_E': []},
            'ZF': {'R_s': [], 'E_I': [], 'R_I': [], 'R_E': []},
            'MRT': {'R_s': [], 'E_I': [], 'R_I': [], 'R_E': []}
        }
        self.P_max_dBm_values = [18, 23, 28, 33, 38, 43]

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # ===== LEFT COLUMN: Controls and Log =====
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Title
        title_label = ttk.Label(left_frame, text="Power Sweep GUI",
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=10)

        # Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        row = 0
        ttk.Label(params_frame, text="M (AP Antennas):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.M_var = tk.IntVar(value=32)
        ttk.Entry(params_frame, textvariable=self.M_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=5)

        row += 1
        ttk.Label(params_frame, text="N (IRS Elements):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.N_var = tk.IntVar(value=50)
        ttk.Entry(params_frame, textvariable=self.N_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=5)

        row += 1
        ttk.Label(params_frame, text="Samples per power:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.num_samples_var = tk.IntVar(value=20)
        ttk.Entry(params_frame, textvariable=self.num_samples_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=5)

        row += 1
        ttk.Label(params_frame, text="AO Trials:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ao_trials_var = tk.IntVar(value=3)
        ttk.Entry(params_frame, textvariable=self.ao_trials_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=5)

        row += 1
        ttk.Label(params_frame, text="AO Max Iter:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ao_max_iter_var = tk.IntVar(value=10)
        ttk.Entry(params_frame, textvariable=self.ao_max_iter_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=5)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(left_frame, text="Select Algorithms", padding="10")
        algo_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        self.algo_ao_var = tk.BooleanVar(value=True)
        self.algo_pso_var = tk.BooleanVar(value=True)
        self.algo_zf_var = tk.BooleanVar(value=True)
        self.algo_mrt_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(algo_frame, text="AO (Alternating Optimization)", variable=self.algo_ao_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(algo_frame, text="PSO (Particle Swarm)", variable=self.algo_pso_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(algo_frame, text="ZF (Zero Forcing)", variable=self.algo_zf_var).grid(row=2, column=0, sticky=tk.W)
        ttk.Checkbutton(algo_frame, text="MRT (Maximum Ratio)", variable=self.algo_mrt_var).grid(row=3, column=0, sticky=tk.W)

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=3, column=0, pady=10)

        self.run_button = ttk.Button(button_frame, text="Run Power Sweep", command=self.run_sweep)
        self.run_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_sweep, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)

        # Log
        log_frame = ttk.LabelFrame(left_frame, text="Log", padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        left_frame.rowconfigure(5, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=45)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=6, column=0, sticky=(tk.W, tk.E))

        # ===== RIGHT COLUMN: Plot Area =====
        right_frame = ttk.LabelFrame(main_frame, text="Results - Power Sweep Comparison", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create matplotlib figure
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=3.0)

        # Initial empty plot
        self.update_plot()

    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_plot(self):
        """Update embedded plot"""
        # Clear all subplots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Get selected algorithms
        selected_algos = []
        if self.algo_ao_var.get():
            selected_algos.append('AO')
        if self.algo_pso_var.get():
            selected_algos.append('PSO')
        if self.algo_zf_var.get():
            selected_algos.append('ZF')
        if self.algo_mrt_var.get():
            selected_algos.append('MRT')

        colors = {'AO': 'red', 'PSO': 'purple', 'ZF': 'green', 'MRT': 'blue'}
        markers = {'AO': 'o', 'PSO': 's', 'ZF': '^', 'MRT': 'd'}

        # Plot 1: Secrecy Rate
        for alg in selected_algos:
            if len(self.results[alg]['R_s']) > 0:
                x_vals = self.P_max_dBm_values[:len(self.results[alg]['R_s'])]
                self.ax1.plot(x_vals, self.results[alg]['R_s'],
                            marker=markers[alg], color=colors[alg],
                            linewidth=2, markersize=8, label=alg)
        self.ax1.set_xlabel('Transmit Power (dBm)', fontsize=10)
        self.ax1.set_ylabel('Secrecy Rate (bps/Hz)', fontsize=10)
        self.ax1.set_title('Secrecy Rate vs Power', fontsize=11, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(fontsize=9)

        # Plot 2: Energy Harvesting
        for alg in selected_algos:
            if len(self.results[alg]['E_I']) > 0:
                x_vals = self.P_max_dBm_values[:len(self.results[alg]['E_I'])]
                self.ax2.plot(x_vals, self.results[alg]['E_I'],
                            marker=markers[alg], color=colors[alg],
                            linewidth=2, markersize=8, label=alg)
        self.ax2.set_xlabel('Transmit Power (dBm)', fontsize=10)
        self.ax2.set_ylabel('Energy Harvesting (dBm)', fontsize=10)
        self.ax2.set_title('Energy Harvesting vs Power', fontsize=11, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(fontsize=9)

        # Plot 3: Information Rate
        for alg in selected_algos:
            if len(self.results[alg]['R_I']) > 0:
                x_vals = self.P_max_dBm_values[:len(self.results[alg]['R_I'])]
                self.ax3.plot(x_vals, self.results[alg]['R_I'],
                            marker=markers[alg], color=colors[alg],
                            linewidth=2, markersize=8, label=alg)
        self.ax3.set_xlabel('Transmit Power (dBm)', fontsize=10)
        self.ax3.set_ylabel('Information Rate (bps/Hz)', fontsize=10)
        self.ax3.set_title('Information Rate vs Power', fontsize=11, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(fontsize=9)

        # Plot 4: Eavesdropper Rate
        for alg in selected_algos:
            if len(self.results[alg]['R_E']) > 0:
                x_vals = self.P_max_dBm_values[:len(self.results[alg]['R_E'])]
                self.ax4.plot(x_vals, self.results[alg]['R_E'],
                            marker=markers[alg], color=colors[alg],
                            linewidth=2, markersize=8, label=alg)
        self.ax4.set_xlabel('Transmit Power (dBm)', fontsize=10)
        self.ax4.set_ylabel('Eavesdropper Rate (bps/Hz)', fontsize=10)
        self.ax4.set_title('Eavesdropper Rate vs Power', fontsize=11, fontweight='bold')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(fontsize=9)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def stop_sweep(self):
        """Stop the running sweep"""
        self.running = False
        self.status_var.set("Stopped by user")

    def run_sweep(self):
        """Run power sweep in a separate thread"""
        if self.running:
            messagebox.showwarning("Warning", "Sweep is already running!")
            return

        # Start in separate thread
        thread = threading.Thread(target=self._run_sweep_thread)
        thread.daemon = True
        thread.start()

    def _run_sweep_thread(self):
        """Thread function to run the power sweep"""
        try:
            self.running = True
            self.run_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.progress.start()

            # Get parameters
            M = self.M_var.get()
            N = self.N_var.get()
            num_samples = self.num_samples_var.get()
            ao_trials = self.ao_trials_var.get()
            ao_max_iter = self.ao_max_iter_var.get()

            # Get selected algorithms
            run_ao = self.algo_ao_var.get()
            run_pso = self.algo_pso_var.get()
            run_zf = self.algo_zf_var.get()
            run_mrt = self.algo_mrt_var.get()

            self.log("="*60)
            self.log("POWER SWEEP STARTED")
            self.log("="*60)
            self.log(f"Config: M={M}, N={N}, Samples={num_samples}")
            self.log(f"Algorithms: AO={run_ao}, PSO={run_pso}, ZF={run_zf}, MRT={run_mrt}")
            self.log(f"Power levels: {self.P_max_dBm_values} dBm")
            self.log(f"CVXPY: {CVXPY_AVAILABLE}")
            self.log("")

            # Fixed parameters
            rho = 0.5
            eta = 0.8
            sigma2 = 1e-8
            K_RICIAN = 10
            radius = 1.5

            # Locations
            BS_loc = np.array([0, 5])
            RIS_loc = np.array([5, 10])
            IU_center = np.array([10, 5])
            EU_center = np.array([5, 0])

            # Initialize baseline
            baseline = BaselineAlgorithms(M, N, rho, eta, sigma2)

            # Clear previous results
            for alg in self.results:
                for key in self.results[alg]:
                    self.results[alg][key] = []

            # Power sweep
            P_max_values = [10**((P_dBm - 30) / 10) for P_dBm in self.P_max_dBm_values]

            for P_idx, (P_max, P_max_dBm) in enumerate(zip(P_max_values, self.P_max_dBm_values)):
                if not self.running:
                    self.log("Stopped by user!")
                    return

                self.log(f"\n--- Power Level {P_idx+1}/6: {P_max_dBm} dBm ---")
                self.status_var.set(f"Processing {P_max_dBm} dBm...")

                # Initialize metrics
                metrics = {
                    'AO': {'R_s': 0, 'R_I': 0, 'R_E': 0, 'E_I': 0},
                    'PSO': {'R_s': 0, 'R_I': 0, 'R_E': 0, 'E_I': 0},
                    'ZF': {'R_s': 0, 'R_I': 0, 'R_E': 0, 'E_I': 0},
                    'MRT': {'R_s': 0, 'R_I': 0, 'R_E': 0, 'E_I': 0}
                }

                # Test on samples
                for sample_idx in range(num_samples):
                    if not self.running:
                        return

                    # Generate channels
                    channels = generate_single_user_channels(BS_loc, RIS_loc, IU_center,
                                                            EU_center, radius, K_RICIAN, M, N)

                    # Run selected algorithms
                    if run_ao:
                        m = baseline.alternating_optimization(channels, P_max, ao_max_iter, ao_trials, 50)
                        for key in metrics['AO']:
                            metrics['AO'][key] += m[key]

                    if run_pso:
                        m = baseline.pso_baseline(channels, P_max, 20, 20)
                        for key in metrics['PSO']:
                            metrics['PSO'][key] += m[key]

                    if run_zf:
                        m = baseline.zero_forcing_baseline(channels, P_max, 5)
                        for key in metrics['ZF']:
                            metrics['ZF'][key] += m[key]

                    if run_mrt:
                        m = baseline.mrt_baseline(channels, P_max)
                        for key in metrics['MRT']:
                            metrics['MRT'][key] += m[key]

                    if (sample_idx + 1) % 5 == 0:
                        self.log(f"  Progress: {sample_idx+1}/{num_samples} samples")

                # Average and store results
                for alg in ['AO', 'PSO', 'ZF', 'MRT']:
                    for key in metrics[alg]:
                        metrics[alg][key] /= num_samples

                    self.results[alg]['R_s'].append(metrics[alg]['R_s'])
                    self.results[alg]['E_I'].append(metrics[alg]['E_I'])
                    self.results[alg]['R_I'].append(metrics[alg]['R_I'])
                    self.results[alg]['R_E'].append(metrics[alg]['R_E'])

                # Log results
                self.log(f"  Results at {P_max_dBm} dBm:")
                if run_ao:
                    self.log(f"    AO:  R_s={metrics['AO']['R_s']:.3f}, E_I={metrics['AO']['E_I']:.2f} dBm")
                if run_pso:
                    self.log(f"    PSO: R_s={metrics['PSO']['R_s']:.3f}, E_I={metrics['PSO']['E_I']:.2f} dBm")
                if run_zf:
                    self.log(f"    ZF:  R_s={metrics['ZF']['R_s']:.3f}, E_I={metrics['ZF']['E_I']:.2f} dBm")
                if run_mrt:
                    self.log(f"    MRT: R_s={metrics['MRT']['R_s']:.3f}, E_I={metrics['MRT']['E_I']:.2f} dBm")

                # Update plot after each power level
                self.update_plot()

            self.log("\n" + "="*60)
            self.log("✅ POWER SWEEP COMPLETE!")
            self.log("="*60)
            self.status_var.set("Complete!")
            messagebox.showinfo("Success", "Power sweep completed!")

        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")

        finally:
            self.running = False
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.progress.stop()

# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    app = PowerSweepGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
