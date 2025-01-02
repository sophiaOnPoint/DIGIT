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

# Ensuring reproducibility of results by fixing the random seed
np.random.seed(10)
# Base vectors (assume known)

# Defining base vectors
a = np.array([1, 0])  # Example base vector a
b = np.array([0, 1])  # Example base vector b, forming a 60-degree angle with a

# Import measured position from experiments
data = loadmat('/home/sophiayd/Dropbox (MIT)/MIT/Research/FDTD/Report/EmitterDiscrete/expMeanPosition.mat')
positions = data['pos']



# Function to compute the position of emitters based on the optimization parameters
def compute_position(best_m_n, theta, U):
    M = best_m_n.shape[0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # best_m_n is Mx2
    assert best_m_n.shape == (M, 2), print(best_m_n.shape)
    lattice_points_in_lattice_space = best_m_n[:, 0:1]* a + best_m_n[:, 1:] * b
    assert lattice_points_in_lattice_space.shape == (M, 2)
    return np.dot(R, lattice_points_in_lattice_space.T).T + U



m_n_range = 2  # Adjust based on expected lattice size000

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
        # m_max = np.round(2/3*(2*np.dot(vector,a)-np.dot(vector,b))).reshape((M,1))
        # n_max = np.round(2/3*(2*np.dot(vector,b)-np.dot(vector,a))).reshape((M,1))

        m_max = np.round(np.dot(vector,a)).reshape((M,1))
        n_max = np.round(np.dot(vector,b)).reshape((M,1))
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
            return compute_position((best_m_n + np.concatenate(((m_max,n_max)),axis=1)),theta,U),(best_m_n + np.concatenate(((m_max,n_max)),axis=1))
        return -ll
    
    return log_likelihood,positions,measurements


# Function to perform the optimization sweep, finding the best parameters that match the simulated measurements
def perform_sweep(M,sigma,seed_emitter = False, seed_noise = False):

    log_likelihood,position_labspace,measurements = build_log_likelihood_function(M, sigma, seed_emitter=seed_emitter, seed_noise=seed_noise)

    #Initiate optimization
    initial_guess = [14/180*np.pi, 0, 0]  # Initial guess for theta, u_x, u_y. Theta is a rough estimation from QR reference
        # Constraint for Uy
    def constraint_uy(params):
        _, ux, uy = params
        return np.abs(ux)/np.sqrt(3) - np.abs(uy)

    constraints = [{'type': 'ineq', 'fun': constraint_uy}]  # Constraints setup
    bounds = [(0, np.pi/2), (0, 1), (0, 1)]  # Bounds for theta, Ux
    result = minimize(log_likelihood, initial_guess, method = 'SLSQP',bounds = bounds,constraints=constraints)
    predicted_positions,return_mn = log_likelihood(result.x,return_predicted_positions=True)
    return result, predicted_positions,return_mn, position_labspace,measurements




## Multi processing to accelerates
def function_wrapper(param):
  M, sigma = param
  
  return perform_sweep(M, sigma, seed_emitter=True, seed_noise=False)


# run for experimental data
data = loadmat('/home/sophiayd/Dropbox (MIT)/MIT/Research/FDTD/Report/SuperRes_pqh/Lattice_experiments/xySTD_QR64_46.mat') #Import measured sigma from SMLM
base = 0.3567
sigma_values = data['sigma']/base
sigma_sweepNumber = sigma_values.size
N = 100
M_values = [positions.shape[0]]*N; 
if __name__ == "__main__":
    N_processes = 30
    pool = Pool(N_processes)

    
    param_pairs = [[M, sigma] for M in M_values for sigma in sigma_values]
    # print(param_pairs)
    t1 = time.time()
    print(f"Running {len(param_pairs)} parameter pairs on {N_processes} processes. Startnig at {datetime.now()}")
    ret = pool.map(function_wrapper, param_pairs)
    t2 = time.time()
    print(f"Finished in {t2-t1} seconds at {datetime.now()}")
    print(f"simulation done")


    N_params = len(param_pairs)
    fitted_positions = np.zeros([sigma_sweepNumber,len(M_values),M_values[0],2])
    original_measurements = np.zeros([sigma_sweepNumber,len(M_values),M_values[0],2])
    fitted_mn = np.zeros([sigma_sweepNumber,len(M_values),M_values[0],2])
    result_optimized = np.zeros([sigma_sweepNumber,len(M_values),3])

    for _i in range(N_params):
        sigma_index = _i % len(sigma_values)
        N_index = _i // len(sigma_values)
        res = ret[_i]
        fitted_mn[sigma_index, N_index,:, :] = res[2]
        original_positions = res[3]
        fitted_positions[sigma_index, N_index,:, :] = res[1]
        original_measurements[sigma_index, N_index,:, :] = res[4]
        result_optimized[sigma_index,N_index,:] = res[0].x
    np.savez(f"diamond_minimize_xi_sigma_M{M_values[0]}_returnMN.npz", sigma_values=sigma_values, result_optimized = result_optimized,
            M_values=M_values, predicted_positions=fitted_positions,original_positions = original_positions, original_measurements = original_measurements, predicted_MN = fitted_mn)





# This script is designed for simulating and optimizing the positioning of quantum emitters on a 2D lattice using experimental data, numerical optimization, and parallel computing. It utilizes principles from physics to estimate emitter locations based on experimental measurements of emitter positions and noise levels. Here's a breakdown of its key functions:

#     Experimental Data Loading:
#         The script loads the experimental data from .mat files, which include the actual positions of the emitters (positions) and corresponding noise levels (sigma).
#         This data is crucial for determining the transformation parameters that best align the predicted emitter locations with the actual experimental measurements.

#     Position Calculation:
#         The compute_position function calculates emitter positions in lab space based on lattice coordinates, a rotation matrix, and an offset vector. This function helps project emitter locations from the lattice space to the experimental lab space.

#     Matching Emitters to Lattice:
#         The match_to_lattice function matches the experimental emitter positions to the closest points on the lattice. It calculates distances between the experimental data and the lattice structure to identify the most likely lattice indices for each emitter.

#     Log-Likelihood Function:
#         The script builds a log-likelihood function using the experimental emitter positions and noise data. This function evaluates how well a set of guessed transformation parameters (rotation theta and offset U) fit the experimental measurements. It compares predicted emitter positions (computed based on lattice points) with the actual experimental positions.

#     Optimization:
#         The perform_sweep function optimizes the transformation parameters (theta, U_x, U_y) by minimizing the difference between predicted and actual positions. The optimization process leverages SciPy’s minimize method with constraints and bounds on the parameters.
#         The goal is to find the best-fit parameters that align the predicted lattice points with the experimental positions.

#     Parallelization:
#         To efficiently handle a large number of emitter positions and varying noise levels (sigma), the script uses Python’s multiprocessing.Pool. This parallelizes the optimization process across different combinations of emitter count M and noise level sigma, significantly speeding up computations.

#     Simulation for Comparison:
#         In addition to experimental data, the script includes commented-out sections that perform simulations, allowing the comparison of optimization results with simulated data by generating noise-perturbed emitter positions.

#     Result Storage:
#         After optimization, the results are saved in .npz files. These results include optimized parameters (theta, U_x, U_y), predicted positions, original experimental positions, and matched lattice indices for further analysis.

# In summary, the script primarily processes experimental data by optimizing the transformation parameters that best align the quantum emitter positions in a 2D grid. The use of parallel computing accelerates the processing of multiple noise levels and emitter positions, while the log-likelihood function ensures accurate alignment between predicted lattice points and experimental measurements.