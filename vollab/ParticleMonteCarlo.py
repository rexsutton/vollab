"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Calculate the prices of european call options using a particle approach,
        with Nadaraya-Watson kernel density estimate.

"""
import numpy as np

from scipy.interpolate import CubicSpline
from scipy import stats


class Delta(object): #pylint: disable=too-few-public-methods
    """
    A functor for applying Nadaraya-Watson kernel density estimation approach,
        specialized for the particle calibration application.
    """

    def __init__(self,  #pylint: disable=too-many-arguments
                 num_particles,
                 spot,
                 time,
                 sigma=1.0,
                 kappa=1.5,
                 time_floor=0.25):
        """
        Simple initialisation.

        Args:
            num_particles: The number of particles being used.
            spot: The current price.
            time: The current times.
            sigma: Estimate of the regressor standard deviation.
            kappa: Scaling factor.
            time_floor: Floor on time.
        """
        self.bandwidth = kappa * spot * sigma * np.sqrt(
            time if time > time_floor else time_floor) / (num_particles ** 0.2)

    @staticmethod
    def _kernel(value):
        """
        Compute the kernel
        .
        Args:
            value: The difference between points.

        Returns:
            The kernel.
        """
        value = 0.0
        if abs(value) < 1.0:
            temp = 1.0 - value * value
            value = (15.0/16.0) * temp * temp
        return value

    def __call__(self, value):
        """
        Compute delta in Nadaraya-Watson kernel density estimation approach.

        Args:
            value: The difference between points.

        Returns:
            The delta value.
        """
        return self._kernel(value / self.bandwidth) / self.bandwidth


class SilvermanDelta(object): #pylint: disable=too-few-public-methods
    """
    Functor for using Silverman's rule of thumb in the Nadaraya-Watson kernel density estimator.
    The specialised delta above appears to out-perform this method.
    """

    def __init__(self, sample_regressor):
        """
        Initialise with sample.

        Args:
            sample_regressor: The sample regressor.
        """
        self.bandwidth = 1.06 * np.std(sample_regressor) / (len(sample_regressor) ** 0.2)

    @staticmethod
    def _kernel(value):
        value = 0.0
        if abs(value) < 1.0:
            temp = 1.0 - value * value
            value = (15.0/16.0) * temp * temp
        return value

    def __call__(self, value):
        return self._kernel(value / self.bandwidth) / self.bandwidth


def expected_volatility_kernel(forward_points,
                               particle_forwards,
                               particle_stochastic_variance):
    """
    Estimate the expected volatility using scipy kernel regression.

    Args:
        forward_points: The forwards.
        particle_forwards: The particle forwards.
        particle_stochastic_variance: The particle variance.

    Returns: Expected volatility as a function of forward.

    """
    data = np.vstack((particle_forwards, particle_stochastic_variance))
    kernel = stats.gaussian_kde(data)
    sigma = np.empty([len(forward_points)])
    for index, forward in enumerate(forward_points):
        k = np.array([kernel([forward, variance]) for variance in particle_stochastic_variance])
        numerator = np.array([density * variance
                              for density, variance in zip(k, particle_stochastic_variance)]).sum()
        sigma[index] = np.sqrt(numerator / k.sum())

    return CubicSpline(forward_points, sigma)


def expected_volatility(delta_func,
                        forward_points,
                        particle_forwards,
                        particle_stochastic_variance):
    """
    Calculate the (inverse of) effective volatility for the forward points.

    Args:
        delta_func: The delta function.
        forward_points: The points at which to estimate effective volatility.
        particle_forwards: The knot points at which the expectation is estimated.
        particle_stochastic_variance: The particle stochastic variance.

    Returns: The inverse effective volatility calculated for the forward points.

    """
    ret = np.ones([len(forward_points)])
    for index, forward in enumerate(forward_points):
        delta_array = np.array([delta_func(f - forward) for f in particle_forwards])
        delta_sum = delta_array.sum()
        variance_sum = np.array(
            [delta * variance
             for delta, variance in zip(delta_array, particle_stochastic_variance)]).sum()
        ret[index] = np.sqrt(variance_sum / delta_sum)
    return ret


class ParticleMonteCarlo(object): #pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Monte Carlo method for generating samples from a stochastic volatility model
        using the particle method.
    """
    def __init__(self, local_vol_surface, strikes, monte_carlo):
        """
        Simple initialization.

        Args:
            local_vol_surface:The local volatility surface.
            strikes: The strikes.
            monte_carlo: The Monte Carlo method for variance.
        """

        self.local_vol_surface = local_vol_surface
        self.strikes = strikes
        self.monte_carlo = monte_carlo
        self.sqrt_step_size = np.sqrt(monte_carlo.step_size)
        self.initial_log_spot = np.log(monte_carlo.spot)
        self.num_times = 1 + monte_carlo.num_steps
        self.num_particles = monte_carlo.num_paths
        self.initial_local_vol = local_vol_surface(monte_carlo.spot, 0.0)
        self.initial_stochatic_vol = monte_carlo.get_initial_vol()
        self.initial_leverage = self.initial_local_vol / self.initial_stochatic_vol

    def sample(self):
        """
        Sample the Monte Carlo.

        Returns:
            Samples paths of the process and the leverage.

        """
        # run monte-carlo
        variance = self.monte_carlo.sample()[1]
        # norms for the forward may be correlated
        norms = self.monte_carlo.get_norms()
        # loop variables
        log_forward = np.full([self.num_particles], self.initial_log_spot)
        forward = np.empty([self.num_times, self.num_particles])
        forward[0] = np.full([self.num_particles], self.monte_carlo.spot)
        leverage = np.empty([self.num_times, self.num_particles])
        leverage[0] = np.full([self.num_particles], self.initial_leverage)
        # run simulation
        idx_time = 1
        # for each times (not including zero)
        for time in self.monte_carlo.time:
            sigma = leverage[idx_time - 1] * np.sqrt(variance[idx_time-1])
            # update forwards
            log_forward += (-0.5 * sigma * sigma * self.monte_carlo.step_size\
                           + sigma * self.sqrt_step_size * norms[idx_time - 1])
            forward[idx_time][:] = np.exp(log_forward)
            # update leverage
            delta_func = Delta(self.num_particles, self.monte_carlo.spot, time, 0.3)
            expected_vol = expected_volatility(delta_func,
                                               self.strikes,
                                               forward[idx_time],
                                               variance[idx_time])
            local_vol = np.array([self.local_vol_surface(f, time)
                                  for f in self.strikes])
            spline = CubicSpline(self.strikes, local_vol / expected_vol)
            leverage[idx_time][:] = np.array([spline(f) for f in forward[idx_time]])
            # increment times
            idx_time += 1
        # return
        return forward, leverage
