"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause
"""


import numpy as np


class HestonMonteCarlo:  # pylint: disable=too-many-instance-attributes
    """
    Monte Carlo method for generating samples from the Heston stochastic volatility model .
    """

    @staticmethod
    def create_with_constant_time_step(step_size=0.1,
                                       num_steps=50,
                                       num_paths=(4 * 2048),
                                       spot=50.0,
                                       drift=0.0,
                                       nu=0.0174,
                                       lmbda=1.3253,
                                       nu_bar=0.0354,
                                       eta=0.3877,
                                       rho=-0.7165):
        time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        return HestonMonteCarlo(time_axis, num_paths,
                                spot,
                                drift,
                                nu,
                                lmbda,
                                nu_bar,
                                eta,
                                rho)

    def __init__(self,  # pylint: disable=too-many-arguments
                 time_axis,
                 num_paths=(4 * 2048),
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
            time_axis: The time axis.
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
            nu: The initial variance.
            lmbda: The mean-reversion speed of volatility.
            nu_bar: The long-run volatility.
            eta: The volatility of volatility.
            rho: The correlation between the process and it's volatility.
        """
        self.time_axis = time_axis
        self.num_paths = num_paths
        self.spot = spot
        self.drift = drift
        self.nu = nu  # pylint: disable=invalid-name
        self.lmbda = lmbda
        self.nu_bar = nu_bar
        self.eta = eta
        self.rho = rho
        self.norms = None

    def process_step(self, log_value, nu, norm, time_step):  # pylint: disable=invalid-name
        """
        Return the updated process value.

        Args:
            log_value: The current value of the process.
            nu: The current variance.
            norm: A standard random normal variate.
            time_step: The size of the times step.

        Returns:
            The updated process value.

        """
        return log_value + (self.drift - 0.5 * nu) * time_step + np.sqrt(nu * time_step) * norm

    def get_norms(self):
        """
        Get the vector of standard random normals used for each variance step.

        Returns:
            Normals used for process step.

        """
        return self.norms[:, 0].reshape([self.num_steps, self.num_paths])

    def get_initial_vol(self):
        """
        Get the initial volatility.

        Returns:
            The initial volatility.
        """
        return np.sqrt(self.nu)

    def variance_step(self, nu, norm, time_step):  # pylint: disable=invalid-name
        """
            Return the updated variance using Milstein method and reflecting condition.

        Args:
            nu: The variance.
            norm: A standard random normal variate.
            time_step: The size of the times step.

        Returns:
            The updated variance.
        """

        temp1 = np.sqrt(nu) + 0.5 * self.eta * np.sqrt(time_step) * norm
        temp2 = -1.0 * self.lmbda * (nu - self.nu_bar) * time_step
        temp3 = -0.25 * self.eta * self.eta * time_step
        nu_new = temp1 * temp1 + temp2 + temp3
        return np.where(np.less(nu_new, np.zeros([len(nu_new)])), -nu_new, nu_new)

    def sample(self):
        """
        Sample the Heston model.

        Returns:
            Samples paths of the process and the variance.
        """
        num_times = len(self.time_axis)
        num_steps = num_times - 1

        means = np.zeros([2])
        covs = np.zeros([2, 2])
        covs[0, 0] = 1.0
        covs[1, 1] = 1.0
        covs[0, 1] = self.rho
        covs[1, 0] = self.rho
        self.norms = np.random.multivariate_normal(means, covs, self.num_paths * num_steps)
        norms1 = self.norms[:, 1].reshape([num_steps, self.num_paths])
        norms2 = self.norms[:, 0].reshape([num_steps, self.num_paths])

        log_process = np.full([self.num_paths], np.log(self.spot))
        process = np.empty([num_times, self.num_paths])
        process[0, :] = self.spot
        variance = np.empty([num_times, self.num_paths])
        variance[0, :] = self.nu

        time_steps = np.diff(self.time_axis)

        for idx, time_step in enumerate(time_steps):
            log_process = self.process_step(log_process,
                                            variance[idx],
                                            norms2[idx],
                                            time_step)
            variance[1 + idx, :] = self.variance_step(variance[idx],
                                                      norms1[idx],
                                                      time_step)
            process[1 + idx, :] = np.exp(log_process)

        return process, variance
