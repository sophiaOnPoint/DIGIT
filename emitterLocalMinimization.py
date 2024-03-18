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

# Ensuring reproducibility of results by fixing the random seed
np.random.seed(10)
# Base vectors (assume known)

# Defining base vectors
a = np.array([np.sqrt(3)/2, 1/2])  # Example base vector a
b = np.array([np.sqrt(3)/2, -1/2])  # Example base vector b, forming a 60-degree angle with a

# Defining true rotation angle theta and offset U for simulation purposes
theta_true = np.pi/12  # 30 degrees
U_true = np.array([0.5, 0.2])

# Simulating measurements with true transformation T(theta_true) and offset U
T_true = np.array([[np.cos(theta_true), -np.sin(theta_true)], [np.sin(theta_true), np.cos(theta_true)]])


# Setting up a grid for simulation
value_range = 10000 # Range of values for the grid
grids_X, grids_Y = np.meshgrid(np.arange(1, value_range + 1), np.arange(1, value_range + 1))

all_pairs = np.vstack([grids_X.flatten().reshape(1, -1), grids_Y.flatten().reshape(1, -1)]).T

assert all_pairs.shape == (value_range**2, 2), print(all_pairs.shape)


# Function to generate new emitter locations based on grid indices and simulate measurements
def generate_new_emitter_location(M):

    assert M <= all_pairs.shape[0], print(M, all_pairs.shape[0])

    indices = np.arange(all_pairs.shape[0])

    np.random.shuffle(indices) # Select M unique pairs

    selected_pairs = indices[:M]# Convert to a NumPy array

    assert selected_pairs.shape == (M,), print(selected_pairs.shape)

    grid_indices = all_pairs[selected_pairs, :]

    assert grid_indices.shape == (M, 2), print(grid_indices.shape)

    p = grid_indices[:, 0:1] * a + grid_indices[:, 1:2] * b

    # Simulate measurements with true T(theta_true) and U

    positions_labspace = np.einsum("ij, lj->li", T_true, p) + U_true

    assert positions_labspace.shape == (M, 2), print(positions_labspace.shape)

    return positions_labspace

# Function to compute the position of emitters based on the optimization parameters
def compute_position(best_m_n, theta, U):
    M = best_m_n.shape[0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # best_m_n is Mx2
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    lattice_points_in_lattice_space = best_m_n[:, 0:1]* a + best_m_n[:, 1:] * b
    assert lattice_points_in_lattice_space.shape == (M, 2)
    return np.dot(R, lattice_points_in_lattice_space.T).T + U



m_n_range = 2  # Adjust based on expected lattice size

Ms, Ns = np.meshgrid(np.arange(-m_n_range, m_n_range + 1), np.arange(-m_n_range, m_n_range + 1))
Ms = Ms.flatten().reshape(-1, 1)
Ns = Ns.flatten().reshape(-1, 1)
lattice_positions = Ms * a + Ns * b
lattice_positions = lattice_positions.reshape(1, 2*m_n_range+1, 2*m_n_range+1, 2)
assert lattice_positions.shape == (1, 2*m_n_range+1, 2*m_n_range+1, 2)

# Function to match the computed positions to a predefined lattice structure
def match_to_lattice(theta, U, positions):
    M = positions.shape[0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lattice_positions_in_lab_space = np.einsum("ij,lmnj->lmni", R, lattice_positions) + U
    assert lattice_positions_in_lab_space.shape == (1, 2*m_n_range+1, 2*m_n_range+1, 2)

    # positions is a Mx2 array
    # lattice_positions is a 1x(2*m_n_range+1)x(2*m_n_range+1)x2 array
    # returns a Mx(2*m_n_range+1)x(2*m_n_range+1) array
    distances = np.sum((lattice_positions_in_lab_space - positions.reshape(M, 1, 1, 2))**2, axis=3)
    assert distances.shape == (M, 2*m_n_range+1, 2*m_n_range+1), print(distances.shape)

    # take the minimum around axis 1 and 2
    distances = distances.reshape(M, -1)
    best_m_n = np.argmin(distances, axis=1)
    assert best_m_n.shape == (M,), print(best_m_n.shape)
    best_m, best_n = np.unravel_index(best_m_n, (2*m_n_range+1, 2*m_n_range+1))
    best_m_n = np.array([best_m, best_n]).T
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    
    # mn is within the range of -m_n_range to m_n_range
    # we need to convert it to the actual m and n
    best_m_n = best_m_n - m_n_range
    # check if this is within range
    assert np.all(np.abs(best_m_n) <= m_n_range)
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    return best_m_n

# Builds the log-likelihood function needed for optimization, simulating the positions and measurements
def build_log_likelihood_function(M, sigma, seed_noise=False, seed_emitter=False):
    if seed_emitter:
        np.random.seed(0)
    else:
        np.random.seed(int(time.time()*1000)%(2**32))        
    positions = generate_new_emitter_location(M)

    if seed_noise:
        np.random.seed(0)
    else:
        np.random.seed(int(time.time()*1000)%(2**32))          
    measurements = np.random.normal(positions, sigma, (M, 2))

    # # Modified log-likelihood function that uses match_to_lattice
    def log_likelihood(params, return_predicted_positions = False):
        theta= params[0]
        u_x = params[1]
        u_y = params[2]

        U = np.array([u_x, u_y])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        v_beforeOffset = (measurements - U)
        vector = np.matmul(R.T, v_beforeOffset.T).T
        # vector = np.transpose(vector).reshape(M,N,2)
        m_max = np.round(2/3*(2*np.dot(vector,a)-np.dot(vector,b))).reshape((M,1))
        n_max = np.round(2/3*(2*np.dot(vector,b)-np.dot(vector,a))).reshape((M,1))
        # vector_AroundOrigin = vector - (m_max * np.transpose(a) + n_max * np.transpose(b));
        vector_AroundOrigin = measurements - (np.matmul(R,(m_max * a + n_max * b).T).T);
        assert vector_AroundOrigin.shape == (M, 2)
        #Relocate to the origin to find the max integer of m and n pair
        best_m_n = match_to_lattice(theta, U, vector_AroundOrigin)
        assert best_m_n.shape == (M, 2), print(best_m_n.shape)
        predicted_positions = compute_position(best_m_n, theta, U)
        assert predicted_positions.shape == (M, 2)
        ll = -np.sum((predicted_positions - vector_AroundOrigin)**2)
        if (return_predicted_positions):
            # return (best_m_n + np.concatenate(((m_max,n_max)),axis=1))
            return compute_position((best_m_n + np.concatenate(((m_max,n_max)),axis=1)),theta,U)
        return -ll
    
    return log_likelihood,positions,measurements

### Test if the log-likelihood function is working
M = 10; sigma=0
test_log_likelihood,_,_ = build_log_likelihood_function(M, sigma)
assert np.isclose(test_log_likelihood((theta_true,0.5,0.2)), 0)
assert np.isclose(test_log_likelihood((theta_true,0.5+0.01,0.2)), M*0.01**2)
assert np.isclose(test_log_likelihood((theta_true,0.5,0.2+0.01)), M*0.01**2)
assert not np.isclose(test_log_likelihood((theta_true+0.01,0.5,0.2)), 0)

# Function to perform the optimization sweep, finding the best parameters that match the simulated measurements
def perform_sweep(M,sigma,seed_emitter = False, seed_noise = False):

    log_likelihood,position_labspace,measurements = build_log_likelihood_function(M, sigma, seed_emitter=seed_emitter, seed_noise=seed_noise)

    #Initiate optimization
    initial_guess = [theta_true, 0.5, 0.2]  # Initial guess for theta, u_x, u_y
        # Constraint for Uy
    def constraint_uy(params):
        _, ux, uy = params
        return np.abs(ux)/np.sqrt(3) - np.abs(uy)

    constraints = [{'type': 'ineq', 'fun': constraint_uy}]  # Constraints setup
    bounds = [(0, np.pi/3), (0, np.sqrt(3)/2), (-0.5, 0.5)]  # Bounds for theta, Ux
    result = minimize(log_likelihood, initial_guess, method = 'SLSQP',bounds = bounds,constraints=constraints)
    predicted_positions = log_likelihood(result.x,return_predicted_positions=True)
    return result, predicted_positions,position_labspace,measurements




## Multi processing to accelerates
# def function_wrapper(param):
#   M, sigma = param

#   return perform_sweep(M, sigma, seed_emitter=True, seed_noise=False)

# M_values = [10,100]*3; 
# sigma_sweepNumber = 2
# sigma_values = np.logspace(-2,1,sigma_sweepNumber)
# if __name__ == "__main__":
#     N_processes = 31
#     pool = Pool(N_processes)

    
#     param_pairs = [[M, sigma] for M in M_values for sigma in sigma_values]
#     print(param_pairs)
#     print(M_values)
    # t1 = time.time()
    # print(f"Running {len(param_pairs)} parameter pairs on {N_processes} processes. Startnig at {datetime.now()}")
    # ret = pool.map(function_wrapper, param_pairs)
    # t2 = time.time()
    # print(f"Finished in {t2-t1} seconds at {datetime.now()}")

    #     N_params = len(param_pairs)
#     loglikelihood_functions = np.zeros((len(M_values), len(sigma_values), stepNumber, stepNumber))

#     for _i in range(N_params):
#         sigma_index = _i % len(sigma_values)
#         M_index = _i // len(sigma_values)
#         res = ret[_i]
#         initial_guess = []
#         loglikelihood_functions[M_index, sigma_index, :, :] = res


# Define digital twin parameters
M_value = 100000; 
sigma_sweepNumber = 2
sigma_values = np.logspace(-2,1,sigma_sweepNumber)
N = 100

# Initialize predicted positions
fitted_positions = np.zeros([sigma_sweepNumber,N,M_value,2])
original_measurements = np.zeros([sigma_sweepNumber,N,M_value,2])
t1 = time.time()

# Sweep across different sigma values
for _j, sigma in (enumerate(sigma_values)):
    # print(sigma)
    t3 = time.time()
    # Monte Carlo N times
    for i in range(N):
        # Call the optimization function
        res = perform_sweep(M_value, sigma, seed_emitter=True, seed_noise=False)
        # Unwrap the results in the right orders
        original_positions = res[2]
        fitted_positions[_j,i,:,:] = res[1]
        original_measurements[_j,i,:,:] = res[3]
    t4 = time.time()
    print(t4-t3)
t2 = time.time()
print(t2-t1)
np.savez(f"minimize_xi_sigma_M{M_value}.npz", sigma_values=sigma_values, 
            M_values=M_value, predicted_positions=fitted_positions,original_positions = original_positions, original_measurements = original_measurements)

