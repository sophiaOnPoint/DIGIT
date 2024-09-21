#########################################################################################################################
# This script is designed for simulating and optimizing the positioning of quantum emitter locations in a 2D grid 
# using principles from physics and optimization algorithms. It utilizes NumPy for numerical computations, multiprocessing 
# to parallelize computations, and SciPy for maxlikelihood routines. The goal is to estimate emitter locations based on 
# simulated measurements, incorporating noise and applying constraints to the optimization problem. It demonstrates an 
# application-centric approach to system design, blending physical simulations with computational optimization techniques.
###########################################################################################################################

from multiprocessing import Pool
import numpy as np
import time
from datetime import datetime
from scipy.optimize import minimize
from scipy.io import loadmat
import math

# Ensuring reproducibility by fixing the random seed
np.random.seed(10)

# Defining base vectors for the lattice
a = np.array([1, 0])  # Base vector along the x-axis
b = np.array([0, 1])  # Base vector along the y-axis, forming a 90-degree angle with a

# Defining true rotation angle (theta_true) and offset (U_true) for simulation purposes
theta_true = np.pi / 12  # True rotation angle (30 degrees)
U_true = np.array([0.5, 0.2])  # True offset vector

# Compute the true transformation matrix T for the rotation
T_true = np.array([[np.cos(theta_true), -np.sin(theta_true)], 
                   [np.sin(theta_true), np.cos(theta_true)]])

# Setting up a 2D grid of emitter locations for simulation
value_range = 30  # The grid size for the emitter positions
grids_X, grids_Y = np.meshgrid(np.arange(1, value_range + 1), np.arange(1, value_range + 1))

# Flatten the grid into pairs of (x, y) coordinates
all_pairs = np.vstack([grids_X.flatten().reshape(1, -1), grids_Y.flatten().reshape(1, -1)]).T
assert all_pairs.shape == (value_range**2, 2), print(all_pairs.shape)

# Function to generate new emitter locations based on random selection of grid indices
def generate_new_emitter_location(M):
    # Ensure we are selecting M unique pairs
    assert M <= all_pairs.shape[0], print(M, all_pairs.shape[0])

    # Randomly shuffle the indices and select M pairs
    indices = np.arange(all_pairs.shape[0])
    np.random.shuffle(indices)  # Shuffle the indices
    selected_pairs = indices[:M]  # Select M random indices
    
    # Get the grid coordinates for the selected indices
    grid_indices = all_pairs[selected_pairs, :]
    assert grid_indices.shape == (M, 2), print(grid_indices.shape)

    # Calculate the positions in lattice space based on grid indices
    p = grid_indices[:, 0:1] * a + grid_indices[:, 1:2] * b
    
    # Apply the true transformation (rotation + offset) to get positions in lab space
    positions_labspace = np.einsum("ij, lj->li", T_true, p) + U_true
    assert positions_labspace.shape == (M, 2), print(positions_labspace.shape)

    return positions_labspace

# Function to compute the position of emitters based on the optimization parameters
def compute_position(best_m_n, theta, U):
    M = best_m_n.shape[0]
    # Create rotation matrix based on theta
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Ensure the shape of the input matches Mx2 (m, n lattice indices)
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    
    # Convert lattice indices (m, n) to positions in lattice space using base vectors a and b
    lattice_points_in_lattice_space = best_m_n[:, 0:1] * a + best_m_n[:, 1:] * b
    assert lattice_points_in_lattice_space.shape == (M, 2)
    
    # Apply the rotation matrix and offset to get the positions in lab space
    return np.dot(R, lattice_points_in_lattice_space.T).T + U

# Set up the lattice search range for emitter matching
m_n_range = 2  # Range of the lattice grid to search over

# Create a meshgrid for all possible m and n values within the lattice range
Ms, Ns = np.meshgrid(np.arange(-m_n_range, m_n_range + 1), np.arange(-m_n_range, m_n_range + 1))
Ms = Ms.flatten().reshape(-1, 1)
Ns = Ns.flatten().reshape(-1, 1)

# Calculate lattice positions in lattice space
lattice_positions = Ms * a + Ns * b
lattice_positions = lattice_positions.reshape(1, 2 * m_n_range + 1, 2 * m_n_range + 1, 2)
assert lattice_positions.shape == (1, 2 * m_n_range + 1, 2 * m_n_range + 1, 2)

# Function to match measured emitter positions to the predefined lattice
def match_to_lattice(theta, U, positions):
    M = positions.shape[0]
    
    # Rotation matrix based on theta
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Rotate the lattice positions and apply the offset to get positions in lab space
    lattice_positions_in_lab_space = np.einsum("ij,lmnj->lmni", R, lattice_positions) + U
    assert lattice_positions_in_lab_space.shape == (1, 2 * m_n_range + 1, 2 * m_n_range + 1, 2)

    # Calculate the squared distances between each measured position and all lattice points
    distances = np.sum((lattice_positions_in_lab_space - positions.reshape(M, 1, 1, 2))**2, axis=3)
    assert distances.shape == (M, 2 * m_n_range + 1, 2 * m_n_range + 1), print(distances.shape)

    # Find the indices of the closest lattice points (minimize distance)
    distances = distances.reshape(M, -1)
    best_m_n = np.argmin(distances, axis=1)
    assert best_m_n.shape == (M,), print(best_m_n.shape)
    
    # Convert flat indices back to m and n grid coordinates
    best_m, best_n = np.unravel_index(best_m_n, (2 * m_n_range + 1, 2 * m_n_range + 1))
    best_m_n = np.array([best_m, best_n]).T
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    
    # Adjust the indices to the correct range (-m_n_range to +m_n_range)
    best_m_n = best_m_n - m_n_range
    assert np.all(np.abs(best_m_n) <= m_n_range)
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    
    return best_m_n

# Build the log-likelihood function for optimization
def build_log_likelihood_function(M, sigma, seed_noise=False, seed_emitter=False):
    # Optionally seed the random number generator for reproducibility
    if seed_emitter:
        np.random.seed(0)
    else:
        np.random.seed(int(time.time() * 1000) % (2**32))

    # Generate the new emitter positions in lab space
    positions = generate_new_emitter_location(M)

    # Add noise to the measurements based on sigma
    if seed_noise:
        np.random.seed(0)
    else:
        np.random.seed(int(time.time() * 1000) % (2**32))
    measurements = np.random.normal(positions, sigma, (M, 2))

    # Define the log-likelihood function used for optimization
    def log_likelihood(params, return_predicted_positions=False):
        theta = params[0]
        u_x = params[1]
        u_y = params[2]
        
        # Offset and rotation matrix
        U = np.array([u_x, u_y])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        # Apply the inverse transformation to the measurements
        v_beforeOffset = (measurements - U)
        vector = np.matmul(R.T, v_beforeOffset.T).T

        # Estimate lattice indices m and n by projecting onto the base vectors
        m_max = np.round(np.dot(vector, a)).reshape((M, 1))
        n_max = np.round(np.dot(vector, b)).reshape((M, 1))
        
        # Subtract the predicted positions from the actual positions
        vector_AroundOrigin = measurements - (np.matmul(R, (m_max * a + n_max * b).T).T)
        assert vector_AroundOrigin.shape == (M, 2)
        
        # Match the computed positions to the nearest lattice points
        best_m_n = match_to_lattice(theta, U, vector_AroundOrigin)
        assert best_m_n.shape == (M, 2), print(best_m_n.shape)
        
        # Compute the predicted positions based on the matched lattice points
        predicted_positions = compute_position(best_m_n, theta, U)
        assert predicted_positions.shape == (M, 2)
        
        # Calculate the negative log-likelihood (sum of squared differences)
        ll = -np.sum((predicted_positions - vector_AroundOrigin) ** 2)
        
        # Optionally return the predicted positions
        if return_predicted_positions:
            return compute_position((best_m_n + np.concatenate((m_max, n_max), axis=1)), theta, U), (best_m_n + np.concatenate((m_max, n_max), axis=1))
        return -ll

    return log_likelihood, positions, measurements

# Test the log-likelihood function to verify correctness
M = 10
sigma = 0
test_log_likelihood, _, _ = build_log_likelihood_function(M, sigma)
assert np.isclose(test_log_likelihood((theta_true, 0.5, 0.2)), 0)
assert np.isclose(test_log_likelihood((theta_true, 0.5 + 0.01, 0.2)), M * 0.01 ** 2)
assert np.isclose(test_log_likelihood((theta_true, 0.5, 0.2 + 0.01)), M * 0.01 ** 2)
assert not np.isclose(test_log_likelihood((theta_true + 0.01, 0.5, 0.2)), 0)

# Perform the optimization sweep to find the best parameters matching the measurements
def perform_sweep(M, sigma, seed_emitter=False, seed_noise=False):
    log_likelihood, position_labspace, measurements = build_log_likelihood_function(M, sigma, seed_emitter=seed_emitter, seed_noise=seed_noise)

    # Initial guess for the optimization parameters (theta, u_x, u_y)
    initial_guess = [theta_true, 0.5, 0.2]

    # Constraint to ensure |u_x| / sqrt(3) > |u_y|
    def constraint_uy(params):
        _, ux, uy = params
        return np.abs(ux) / np.sqrt(3) - np.abs(uy)

    # Set up constraints and bounds for the optimization
    constraints = [{'type': 'ineq', 'fun': constraint_uy}]
    bounds = [(0, np.pi / 2), (0, 1), (0, 1)]  # Bounds for theta, u_x, and u_y

    # Perform optimization using the SLSQP method
    result = minimize(log_likelihood, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the predicted positions and matched lattice points
    predicted_positions, return_mn = log_likelihood(result.x, return_predicted_positions=True)
    
    return result, predicted_positions, return_mn, position_labspace, measurements

# Multiprocessing function to accelerate optimization over different parameter combinations
def function_wrapper(param):
    M, sigma = param
    return perform_sweep(M, sigma, seed_emitter=True, seed_noise=False)

# Run the optimization for simulated data with multiple processes
data = loadmat('/home/sophiayd/Dropbox (MIT)/MIT/Research/FDTD/Report/SuperRes_pqh/Lattice_experiments/xySTD_QR64_46.mat')
base = 0.31
sigma_values = data['sigma'] / base / 1.7  # Adjust sigma based on experimental data
sigma_sweepNumber = sigma_values.size  # Number of sigma values to sweep over
N = 10000  # Number of simulations
M_values = [5] * N  # Number of emitters (M)

if __name__ == "__main__":
    N_processes = 30  # Number of parallel processes to run
    pool = Pool(N_processes)

    # Create parameter pairs (M, sigma) for multiprocessing
    param_pairs = [[M, sigma] for M in M_values for sigma in sigma_values]

    # Start the timer and run the parallel optimization
    t1 = time.time()
    print(f"Running {len(param_pairs)} parameter pairs on {N_processes} processes. Starting at {datetime.now()}")
    ret = pool.map(function_wrapper, param_pairs)
    t2 = time.time()
    print(f"Finished in {t2-t1} seconds at {datetime.now()}")
    print(f"Simulation done")

    # Prepare arrays to store the results
    fitted_positions = np.zeros([sigma_sweepNumber, len(M_values), M_values[0], 2])
    original_measurements = np.zeros([sigma_sweepNumber, len(M_values), M_values[0], 2])
    fitted_mn = np.zeros([sigma_sweepNumber, len(M_values), M_values[0], 2])
    result_optimized = np.zeros([sigma_sweepNumber, len(M_values), 3])

    # Collect the results from each process
    N_params = len(param_pairs)
    for _i in range(N_params):
        sigma_index = _i % len(sigma_values)
        N_index = _i // len(sigma_values)
        res = ret[_i]
        fitted_mn[sigma_index, N_index, :, :] = res[2]
        original_positions = res[3]
        fitted_positions[sigma_index, N_index, :, :] = res[1]
        original_measurements[sigma_index, N_index, :, :] = res[4]
        result_optimized[sigma_index, N_index, :] = res[0].x

    # Save the results to a .npz file for further analysis
    np.savez(f"diamond_minimize_xi_sigma_M{M_values[0]}_returnMN_10000Trail.npz", sigma_values=sigma_values, result_optimized=result_optimized,
             M_values=M_values, predicted_positions=fitted_positions, original_positions=original_positions, original_measurements=original_measurements, predicted_MN=fitted_mn)


