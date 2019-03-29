"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2018.

    Monte Carlo method for generating samples from the Heston stochastic volatility model .

"""

import math
import numpy as np
from numba import cuda


@cuda.jit("(float32[:], float32, float32, float32)", inline=True, device=True)
def process_step(params, log_value, nu, norm):  # pylint: disable=invalid-name
    """
    Return the updated process value.

    Args:
        params: The process parameters.
        log_value: The current value of the process.
        nu: The current variance.
        norm: A standard random normal variate.

    Returns:
        The updated process value.

    """
    time_step = params[0]
    drift = params[2]
    return log_value + (drift - 0.5 * nu) * time_step + math.sqrt(nu * time_step) * norm


@cuda.jit("(float32[:], float32, float32)", inline=True, device=True)
def variance_step(params, nu, norm):  # pylint: disable=invalid-name
    """
        Return the updated variance using Milstein method and reflecting condition.

    Args:
        params: The process parameters.
        nu: The variance.
        norm: A standard random normal variate.

    Returns:
        The updated variance.
    """
    time_step = params[0]
    lmbda = params[4]
    nu_bar = params[5]
    eta = params[6]

    temp1 = math.sqrt(nu) + 0.5 * eta * math.sqrt(time_step) * norm
    temp2 = -1.0 * lmbda * (nu - nu_bar) * time_step
    temp3 = -0.25 * eta * eta * time_step
    nu_new = temp1 * temp1 + temp2 + temp3

    if nu_new < 0.0:
        nu_new = -nu_new

    return nu_new


@cuda.jit("(int32, float32[:], float32[:], float32[:], float32[:,:])")
def heston_kernel(num_steps, params, norms1, norms2, output):
    """
    Generate Heston process values in the output array.
    Args:
        num_steps: The number of steps.
        params: The parameters.
        norms1: The random normals for the process.
        norms2: The random normals for the variance.
        output: The output array.

    Returns:

    """
    # thread and blocking organisation
    idx_time = cuda.blockIdx.x
    idx_path = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    # extract params (could be constant memory)
    process = params[1]
    variance = params[3]
    # simulate the path
    idx_start = num_steps * idx_path
    for idx_rand in range(idx_start, idx_start + idx_time):
        process = process_step(params, process, variance, norms1[idx_rand])
        variance = variance_step(params, variance, norms2[idx_rand])
    # store the result
    output[idx_time, idx_path] = math.exp(process)


class CudaHestonMonteCarlo(object): #pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Monte Carlo method for generating samples from the Heston stochastic volatility model .
    """

    def __init__(self, #pylint: disable=too-many-arguments
                 step_size=0.1,
                 num_steps=50,
                 num_paths=(4*2048),
                 spot=50.0,
                 drift=0.0,
                 nu=0.0174,
                 lmbda=1.3253,
                 nu_bar=0.0354,
                 eta=0.3877,
                 rho=-0.7165):
        """
        Member-wise initialisation.

        Args:
            step_size: The Monte Carlo step size in years.
            num_steps: The number of steps.
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
            nu: The initial variance.
            lmbda: The mean-reversion speed of volatility.
            nu_bar: The long-run volatility.
            eta: The volatility of volatility.
            rho: The correlation between the process and it' volatility.
        """
        self.spot = spot
        self.drift = drift
        self.nu = nu # pylint: disable=invalid-name
        self.lmbda = lmbda
        self.nu_bar = nu_bar
        self.eta = eta
        self.rho = rho
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.time = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.norms = None


    def sample(self):
        """

        Sample the Heston model.

        Returns:
            Samples paths of the process and the variance.
        """
        # device properties
        warp_size = 32
        max_threads_per_block = 1024

        # do blocking calculation
        num_times = 1 + self.num_steps
        if self.num_paths == 0:
            raise RuntimeError("Zero paths!")
        if self.num_paths % warp_size != 0:
            raise RuntimeError("Paths is not a multiple of the warp size.")
        num_blocks_y = self.num_paths / max_threads_per_block
        if self.num_paths % max_threads_per_block:
            num_blocks_y += 1
        num_paths_per_block = max_threads_per_block
        num_new_paths = num_paths_per_block * num_blocks_y

        # sanity checking
        if num_paths_per_block < warp_size:
            raise RuntimeError("Can't schedule fewer threads than a single warp.")
        if num_times * num_blocks_y * num_paths_per_block != num_times * num_new_paths:
            raise RuntimeError("Number of calculations does not match the number of threads.")

        # generate random numbers on CPU, could be done on GPU
        means = np.zeros([2])
        covariance_matrix = np.zeros([2, 2])
        covariance_matrix[0, 0] = 1.0
        covariance_matrix[1, 1] = 1.0
        covariance_matrix[0, 1] = self.rho
        covariance_matrix[1, 0] = self.rho
        self.norms = np.random.multivariate_normal(means,
                                                   covariance_matrix,
                                                   num_new_paths * self.num_steps)
        norms1 = np.array(self.norms[:, 1], dtype=np.float32)
        norms2 = np.array(self.norms[:, 0], dtype=np.float32)

        # copy to parameters array
        params = np.array([self.step_size,
                           np.log(self.spot),
                           self.drift,
                           self.nu,
                           self.lmbda,
                           self.nu_bar,
                           self.eta,
                           self.rho], np.float32)

        # make space for results
        output = np.empty([(1+self.num_steps), num_new_paths], dtype=np.float32)

        # invoke the kernel
        heston_kernel[(num_times, num_blocks_y, 1), (num_paths_per_block, 1, 1)](self.num_steps,
                                                                                 params,
                                                                                 norms1,
                                                                                 norms2,
                                                                                 output)
        return output
