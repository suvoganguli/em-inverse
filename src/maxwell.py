import numpy as np

from config import (
    Lx, Ly, Nx_fdtd, Ny_fdtd,
    eps0, mu0, c0, f, omega,
    eps_r_bg, sigma_bg,
    ant_x, ant_y, N_ant,
    dx_fdtd, dy_fdtd, dt, n_steps,
    X_RANGE, Y_RANGE, R_RANGE, C_RANGE, SIGMA_T_RANGE
)


# =========================================================
# GRID + ANTENNA INDEX SETUP
# =========================================================

x_fdtd = np.linspace(-Lx / 2, Lx / 2, Nx_fdtd)
y_fdtd = np.linspace(-Ly / 2, Ly / 2, Ny_fdtd)
Xfdt, Yfdt = np.meshgrid(x_fdtd, y_fdtd)

ant_ix_fdtd = np.array([np.argmin(np.abs(x_fdtd - ax)) for ax in ant_x])
ant_iy_fdtd = np.array([np.argmin(np.abs(y_fdtd - ay)) for ay in ant_y])


# =========================================================
# DAMPING MASK
# =========================================================

def build_damping_mask(n_damp=10, strength=0.03):
    damping = np.ones((Ny_fdtd, Nx_fdtd), dtype=float)

    for i in range(Ny_fdtd):
        for j in range(Nx_fdtd):
            di = min(i, Ny_fdtd - 1 - i)
            dj = min(j, Nx_fdtd - 1 - j)
            dmin = min(di, dj)

            if dmin < n_damp:
                damping[i, j] = 1.0 - strength * (n_damp - dmin) / n_damp

    return damping


DAMPING = build_damping_mask()


# =========================================================
# MATERIAL MAPS
# =========================================================

def build_material_maps(x_t, y_t, r_t, eps_r_t, sigma_t=None):
    """
    Build relative permittivity and conductivity maps for one circular tumor.
    """
    if sigma_t is None:
        sigma_t = sigma_bg

    eps_r_map = eps_r_bg * np.ones((Ny_fdtd, Nx_fdtd), dtype=float)
    sigma_map = sigma_bg * np.ones((Ny_fdtd, Nx_fdtd), dtype=float)

    rr = np.sqrt((Xfdt - x_t) ** 2 + (Yfdt - y_t) ** 2)
    mask = rr <= r_t

    eps_r_map[mask] = eps_r_t
    sigma_map[mask] = sigma_t

    return eps_r_map, sigma_map, mask


def build_background_maps():
    """
    Homogeneous background with no tumor.
    """
    eps_r_map = eps_r_bg * np.ones((Ny_fdtd, Nx_fdtd), dtype=float)
    sigma_map = sigma_bg * np.ones((Ny_fdtd, Nx_fdtd), dtype=float)
    return eps_r_map, sigma_map


# =========================================================
# PHASOR EXTRACTION
# =========================================================

def extract_complex_at_f0(signal, dt_local, f0):
    """
    Extract a complex phasor at frequency f0 from a real-valued time signal.
    """
    n = np.arange(len(signal))
    t = n * dt_local
    kernel = np.exp(-1j * 2 * np.pi * f0 * t)
    return np.sum(signal * kernel)


# =========================================================
# SINGLE-TX FDTD RUN
# =========================================================

def run_fdtd_tx(tx_idx, eps_r_map, sigma_map, return_final_field=False):
    """
    Run one compact TM-mode FDTD simulation for a given TX index.

    Returns
    -------
    rx_signals : ndarray, shape (N_ant, n_steps)
        Time signals recorded at all antenna locations.
    Ez_final : ndarray, optional
        Final Ez field snapshot, only if return_final_field=True.
    """
    # source location
    ix_tx = ant_ix_fdtd[tx_idx]
    iy_tx = ant_iy_fdtd[tx_idx]

    # fields
    Ez = np.zeros((Ny_fdtd, Nx_fdtd), dtype=float)
    Hx = np.zeros((Ny_fdtd - 1, Nx_fdtd), dtype=float)
    Hy = np.zeros((Ny_fdtd, Nx_fdtd - 1), dtype=float)

    # material coefficients
    eps_abs = eps0 * eps_r_map
    Ca = (1 - sigma_map * dt / (2 * eps_abs)) / (1 + sigma_map * dt / (2 * eps_abs))
    Cb = (dt / eps_abs) / (1 + sigma_map * dt / (2 * eps_abs))

    # receiver signals
    rx_signals = np.zeros((N_ant, n_steps), dtype=float)

    for n in range(n_steps):
        # update Hx
        Hx -= (dt / mu0) * (Ez[1:, :] - Ez[:-1, :]) / dy_fdtd

        # update Hy
        Hy += (dt / mu0) * (Ez[:, 1:] - Ez[:, :-1]) / dx_fdtd

        # curl(H)
        curl_H = np.zeros_like(Ez)
        curl_H[1:-1, 1:-1] = (
            (Hy[1:-1, 1:] - Hy[1:-1, :-1]) / dx_fdtd
            - (Hx[1:, 1:-1] - Hx[:-1, 1:-1]) / dy_fdtd
        )

        # update Ez
        Ez = Ca * Ez + Cb * curl_H

        # Gaussian-modulated sinusoidal source
        t = n * dt
        t0 = 40 * dt
        spread = 12 * dt
        source = np.sin(2 * np.pi * f * t) * np.exp(-((t - t0) ** 2) / (2 * spread ** 2))
        Ez[iy_tx, ix_tx] += source

        # damping
        Ez *= DAMPING

        # record receivers
        for rx_idx in range(N_ant):
            rx_signals[rx_idx, n] = Ez[ant_iy_fdtd[rx_idx], ant_ix_fdtd[rx_idx]]

    if return_final_field:
        return rx_signals, Ez

    return rx_signals


# =========================================================
# SINGLE-TX COMPLEX SCATTERED MEASUREMENT
# =========================================================

def simulate_tx_measurement(tx_idx, x_t, y_t, r_t, eps_r_t, sigma_t=None):
    """
    Run tumor-present and background cases for one TX, then form
    complex scattered measurements at all RX antennas.
    """
    eps_r_map, sigma_map, _ = build_material_maps(x_t, y_t, r_t, eps_r_t, sigma_t)
    eps_r_bg_map, sigma_bg_map = build_background_maps()

    rx_tumor = run_fdtd_tx(tx_idx, eps_r_map, sigma_map)
    rx_bg = run_fdtd_tx(tx_idx, eps_r_bg_map, sigma_bg_map)

    rx_scat_time = rx_tumor - rx_bg

    E_scat = np.zeros(N_ant, dtype=complex)
    for rx_idx in range(N_ant):
        E_scat[rx_idx] = extract_complex_at_f0(rx_scat_time[rx_idx], dt, f)

    # zero out self-receiver for consistency
    E_scat[tx_idx] = 0.0 + 0.0j

    return E_scat


# =========================================================
# FULL MULTISTATIC MEASUREMENT MATRIX
# =========================================================

def simulate_multistatic_measurement(x_t, y_t, r_t, eps_r_t, sigma_t=None):
    """
    Generate full multistatic measurement matrix of shape (N_ant, N_ant),
    where rows = TX and columns = RX.
    """
    M = np.zeros((N_ant, N_ant), dtype=complex)

    for tx_idx in range(N_ant):
        M[tx_idx] = simulate_tx_measurement(tx_idx, x_t, y_t, r_t, eps_r_t, sigma_t)

    return M


def complex_matrix_to_tensor(M):
    """
    Convert complex matrix (N_ant, N_ant) to real tensor (N_ant, N_ant, 2)
    with channels [real, imag].
    """
    X = np.zeros((N_ant, N_ant, 2), dtype=np.float32)
    X[..., 0] = np.real(M)
    X[..., 1] = np.imag(M)
    return X


# =========================================================
# RANDOM PARAMETER SAMPLING
# =========================================================

def sample_xy(r_fixed=0.01, c_fixed=50.0, sigma_fixed=4.0, rng=None):
    """
    Sample only x, y. Keep radius and contrast fixed.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_t = rng.uniform(*X_RANGE)
    y_t = rng.uniform(*Y_RANGE)

    return x_t, y_t, r_fixed, c_fixed, sigma_fixed


def sample_xyr(c_fixed=50.0, sigma_fixed=4.0, rng=None):
    """
    Sample x, y, r. Keep contrast fixed.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_t = rng.uniform(*X_RANGE)
    y_t = rng.uniform(*Y_RANGE)
    r_t = rng.uniform(*R_RANGE)

    return x_t, y_t, r_t, c_fixed, sigma_fixed


def sample_xyrc(rng=None):
    """
    Sample x, y, r, c. Also sample sigma_t in a simple correlated way.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_t = rng.uniform(*X_RANGE)
    y_t = rng.uniform(*Y_RANGE)
    r_t = rng.uniform(*R_RANGE)
    c_t = rng.uniform(*C_RANGE)
    sigma_t = rng.uniform(*SIGMA_T_RANGE)

    return x_t, y_t, r_t, c_t, sigma_t


# =========================================================
# DATASET GENERATION
# =========================================================

def generate_dataset_xy(n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_data = np.zeros((n_samples, N_ant, N_ant, 2), dtype=np.float32)
    y_data = np.zeros((n_samples, 2), dtype=np.float32)

    for i in range(n_samples):
        x_t, y_t, r_t, c_t, sigma_t = sample_xy(rng=rng)
        M = simulate_multistatic_measurement(x_t, y_t, r_t, c_t, sigma_t)

        X_data[i] = complex_matrix_to_tensor(M)
        y_data[i] = np.array([x_t, y_t], dtype=np.float32)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[xy] generated {i + 1}/{n_samples}")

    return X_data, y_data


def generate_dataset_xyr(n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_data = np.zeros((n_samples, N_ant, N_ant, 2), dtype=np.float32)
    y_data = np.zeros((n_samples, 3), dtype=np.float32)

    for i in range(n_samples):
        x_t, y_t, r_t, c_t, sigma_t = sample_xyr(rng=rng)
        M = simulate_multistatic_measurement(x_t, y_t, r_t, c_t, sigma_t)

        X_data[i] = complex_matrix_to_tensor(M)
        y_data[i] = np.array([x_t, y_t, r_t], dtype=np.float32)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[xyr] generated {i + 1}/{n_samples}")

    return X_data, y_data


def generate_dataset_xyrc(n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_data = np.zeros((n_samples, N_ant, N_ant, 2), dtype=np.float32)
    y_data = np.zeros((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        x_t, y_t, r_t, c_t, sigma_t = sample_xyrc(rng=rng)
        M = simulate_multistatic_measurement(x_t, y_t, r_t, c_t, sigma_t)

        X_data[i] = complex_matrix_to_tensor(M)
        y_data[i] = np.array([x_t, y_t, r_t, c_t], dtype=np.float32)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[xyrc] generated {i + 1}/{n_samples}")

    return X_data, y_data
