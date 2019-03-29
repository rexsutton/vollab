#!/usr/bin/env python
"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Demo particle Monte Carlo calibration using TensorFlow.

"""

import argparse
import functools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import TensorFlowCubicSpline as tfcs
import env
import vollab as vl


def plot_surface(x_axis, y_axis, z_values, tol=1e-8):
    """
        Plot a surface using MatPlotLib.
        Tolerance controls what appears to be a bug when plotting flat surfaces.
    Args:
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        tol: Tolerance for rounding numbers to precision.
    """
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing='ij')
    z_matrix = np.zeros([len(x_axis), len(y_axis)], dtype=np.float64)

    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            z_matrix[i, j] = np.floor(z_values[i][j] / tol)*tol

    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_matrix)
    sub.title.set_text("Call prices as a function of strike and maturity.")
    sub.set_xlabel("Strike")
    sub.set_ylabel("Maturity")
    sub.set_zlabel("Price")


def setup_local_vol(model, params):
    """
    For the specified model compute plain vanilla option prices obtained via FFT,
     and the local volatility surface.

    Args:
        model: A string identifying the model's characteristic function.
        params: A dictionary of parameter values used to override defaults.
    """
    # create market parameters
    market_params = vl.MarketParams()
    market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function(model)
    characteristic_function.__dict__.update(params)
    # select the range of strikes to work with
    strike_selector = functools.partial(vl.select_strike,
                                        0.7 * market_params.spot,
                                        1.3 * market_params.spot)
    # generate a surface out to five years
    tenors = np.linspace(0.0, 50 * 0.1, 1 + 50)
    # calculate call prices
    selected_strikes, tenors, call_prices_by_fft = \
        vl.compute_call_prices_matrix(characteristic_function,
                                      market_params,
                                      strike_selector,
                                      tenors)
    # calculate the local vol surface
    floored_tenors = np.array([tenor for tenor in tenors if tenor > 0.25])
    local_vol_matrix_results \
        = vl.compute_local_vol_matrix(characteristic_function,
                                      market_params,
                                      strike_selector,
                                      floored_tenors)
    # make the spline surface
    local_vol_spline_surface = vl.SplineSurface(selected_strikes,
                                                floored_tenors,
                                                local_vol_matrix_results[2],
                                                tenors)

    # print local_vol_spline_surface(market_params.spot, 0.0)
    return selected_strikes, tenors, call_prices_by_fft, local_vol_spline_surface


class HestonParticleCalibrator(object):  # pylint: disable=too-many-instance-attributes
    """
    Monte Carlo method for generating samples from a stochastic volatility model
        using the particle method.
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 local_vol_surface,
                 strikes,
                 step_size=0.5,
                 num_steps=10,
                 num_paths=4*2048,
                 spot=50.0,
                 nu=0.0174,
                 lmbda=1.3253,
                 nu_bar=0.0354,
                 eta=0.3877,
                 rho=-0.7165):
        """
            Initialisation.
        """
        self.local_vol_surface = local_vol_surface
        self.strikes = strikes
        self.step_size = np.float32(step_size)
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.spot = np.float32(spot)
        self.nu = np.float32(nu)  # pylint: disable=invalid-name
        self.lmbda = np.float32(lmbda)
        self.nu_bar = np.float32(nu_bar)
        self.eta = np.float32(eta)
        self.rho = np.float32(rho)
        # Monte Carlo
        self.time = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.sqrt_step_size = np.float32(np.sqrt(step_size))
        # parameters of the delta function
        self.sigma_variance_swap = np.float32(0.3)
        self.kappa = np.float32(1.5)
        self.time_floor = np.float32(0.25)
        self.r_tensor = tf.constant(np.float32(15.0 / 16.0))
        # strike constant tensors
        self.strikes_tensor = tf.constant(np.array(strikes, np.float32))
        self.strike_tensors = []
        for strike in strikes:
            self.strike_tensors.append(tf.constant(np.full([num_paths], np.float32(strike))))

        self.strike_min_tensor = tf.constant(np.full([num_paths], np.min(strikes), np.float32))
        self.strike_max_tensor = tf.constant(np.full([num_paths], np.max(strikes), np.float32))
        self.log_strike_min_tensor = tf.constant(np.full([num_paths],
                                                         np.log(np.min(strikes)), np.float32))
        self.log_strike_max_tensor = tf.constant(np.full([num_paths],
                                                         np.log(np.max(strikes)), np.float32))
        # initial values
        self.initial_local_vol = np.float32(self.local_vol_surface(self.spot, 0.0))
        self.initial_log_forward_tensor = tf.constant(np.full([num_paths], np.log(self.spot)))
        self.initial_forward_tensor = tf.constant(np.full([num_paths], self.spot))
        self.initial_leverage_tensor = tf.constant(
            np.full([num_paths], self.initial_local_vol / np.sqrt(self.nu)))
        self.initial_variance_tensor = tf.constant(np.full([num_paths], self.nu))
        # useful constant tensors
        self.half = np.float32(0.5)
        self.quarter = np.float32(0.25)
        self.one_tensor = tf.constant(np.float32(1.0))
        self.ones_tensor = tf.constant(np.full([self.num_paths], np.float32(1.0)))
        self.zero_tensor = tf.constant(np.float32(0.0))
        self.zeros_tensor = tf.constant(np.full([self.num_paths], np.float32(0.0)))

    def _compute_bandwidth(self, maturity):
        """
        Calculate the bandwidth used in the kernel density regression.

        Args:
            maturity: The maturity time in years.

        Returns: The bandwidth as a constant tensor.

        """
        bandwidth = self.kappa * self.spot * self.sigma_variance_swap\
            * np.sqrt(np.max([maturity, self.time_floor])) \
            / (self.num_paths ** 0.2)
        return tf.constant(np.float32(bandwidth))

    def _expected_volatility(self, maturity, forwards, variance):

        """
        Calculate the (inverse of) effective volatility for the forward points.

        Args:
            maturity: The maturity used in the bandwidth calculation.
            forwards: The knot points at which the expectation is estimated.
            variance: The particle stochastic variance.

        Returns: The inverse effective volatility calculated for the forward points.
        """
        bandwidth_tensor = self._compute_bandwidth(maturity)
        temp = []
        for strike_tensor in self.strike_tensors:
            difs = (forwards - strike_tensor) / bandwidth_tensor
            delta_array_values = self.r_tensor * tf.square(self.ones_tensor - tf.square(difs))
            cond = tf.less(tf.abs(difs), self.ones_tensor)
            delta_array = tf.where(cond, delta_array_values, self.zeros_tensor) / bandwidth_tensor
            variance_sum = tf.reduce_sum(delta_array * variance)
            temp.append(tf.sqrt(variance_sum / tf.reduce_sum(delta_array)))
        return tf.reshape(tf.stack(temp), [int(len(temp))])

    def _cut_forwards(self, forwards):
        """
        Particle forwards beyond the max strike and below min_strike are returned to the spot.

        Args:
            forwards: The simulated particle forwards.

        Returns: A pair where the forwards were cut and the cut forwards.

        """
        # cut-off particle forwards beyond the max strike and below min_strike
        condition1 = tf.less(forwards, self.strike_max_tensor)
        condition2 = tf.greater(forwards, self.strike_min_tensor)
        condition = tf.logical_and(condition1, condition2)
        return condition, tf.where(condition, forwards, self.initial_forward_tensor)

    def _compute_leverage(self,
                          maturity,
                          forwards,
                          variance):

        """
        Calculate the leverage ratio.

        Args:
            maturity: The maturity leverage is being computed at.
            forwards: The particle forwards.
            variance: The particle variance.

        Returns:

        """
        expected_vol = self._expected_volatility(maturity, forwards, variance)
        local_vol = tf.constant(np.array([self.local_vol_surface(strike, maturity)
                                          for strike in self.strikes], np.float32))
        condition, cut_forwards = self._cut_forwards(forwards)
        spline_values = tfcs.cubic_spline(self.strikes, local_vol / expected_vol, cut_forwards)
        return tf.where(condition, spline_values, self.ones_tensor)

    def _variance_step(self, nu, norm, vol_params, dt):  # pylint: disable=invalid-name
        """
            Return the updated variance using Milstein method and reflecting condition.

        Args:
            nu: The variance.
            norm: A standard random normal variate.
            vol_params: The parameters.
            dt: The size of the time step.

        Returns:
            The updated variance.
        """
        temp1 = tf.sqrt(nu) + self.half * vol_params[2] * np.sqrt(dt) * norm
        temp2 = -vol_params[0] * (nu - vol_params[1]) * dt
        temp3 = -self.quarter * vol_params[2] * vol_params[2] * dt
        nu_new = temp1 * temp1 + temp2 + temp3
        neg_nu_new = -1.0 * nu_new
        return tf.maximum(nu_new, neg_nu_new)

    def _create_sampler(self, vol_params, vary_correlation):
        """
        Create sampler for the random variates.

        Args:
            vol_params: The parameters.
            vary_correlation: If True then correlation is the last parameter.

        Returns:
            The sampler for the random variates.

        """

        if vary_correlation:
            corr_tensor = vol_params[3]
        else:
            corr_tensor = tf.constant(self.rho)

        correl_tensor = tf.stack([tf.stack([self.one_tensor, corr_tensor]),
                                  tf.stack([corr_tensor, self.one_tensor])])
        return tf.contrib.distributions.MultivariateNormalFullCovariance(tf.constant([0.0, 0.0]),
                                                                         correl_tensor)

    def sample(self, vol_params, vary_correlation):
        """
        Sample the Monte Carlo.

        Returns:
            Samples paths of the process.

        """
        # create sampler for the randoms
        dist = self._create_sampler(vol_params, vary_correlation)
        # create the state
        state = self.initial_log_forward_tensor,\
                self.initial_forward_tensor,\
                self.initial_leverage_tensor,\
                self.initial_variance_tensor
        # run simulation
        samples = [state[1]]
        # for each times (not including zero)
        for time in self.time:
            # calculate the volatility
            sigma = state[2] * tf.sqrt(state[3])
            # generate normals
            rands_tensor = dist.sample([self.num_paths])
            # update the forwards
            log_forward = state[0] - 0.5 * sigma * sigma * self.step_size \
                          + sigma * self.sqrt_step_size * rands_tensor[:, 0]
            forwards = tf.exp(log_forward)
            # update the variance
            variance = self._variance_step(state[3], rands_tensor[:, 1], vol_params, self.step_size)
            # update the leverage
            leverage = self._compute_leverage(time, forwards, variance)
            # update the state
            state = log_forward, forwards, leverage, variance
            # append the forwards to the state
            samples.append(state[1])
        # return
        return samples

    def _call_prices(self, forwards_tensor):
        """
            Compute call prices from simulated values.

        Args:
            forwards_tensor: The simulated forwards.

        Returns:
            The call price for each strike.

        """
        lst = []
        for strike_tensor in self.strike_tensors:
            difs = forwards_tensor - strike_tensor
            cond = tf.greater(difs, self.zeros_tensor)
            intrinsic = tf.where(cond, difs, self.zeros_tensor)
            call_price = tf.reduce_mean(intrinsic)
            lst.append(call_price)
        return tf.stack(lst)

    def compute_call_prices(self, vol_params, select_times, vary_correlation):
        """
            Compute call prices from simulated values at the selected times.

        Args:
            vol_params: The parameters of the model.
            vary_correlation: If True vary correlation parameter.
            select_times: Function for selecting the maturities to return.

        Returns:
            The call prices for each strike at the selected times.

        """
        samples = self.sample(vol_params, vary_correlation)
        call_price_surface = []
        times = []
        # for each time
        for time, samples in zip(self.time_axis, samples):
            # update call prices
            if select_times(time):
                times.append(time)
                call_price_surface.append(self._call_prices(samples))
        # return
        return self.strikes, times, tf.stack(call_price_surface)


def test_surface():
    """
    Compute plain vanilla call prices at the strikes, and at selected maturities.
    """
    selected_strikes, _, _, local_vol_spline_surface \
        = setup_local_vol('Heston', dict())
    # create the Monte Carlo
    pmc = HestonParticleCalibrator(local_vol_spline_surface, np.array(selected_strikes, np.float32))
    vol_params = tf.constant(np.array([1.3253, 0.0354, 0.3877, -0.7165], np.float32))
    # use particle method to calculate call prices
    select_times = lambda x: True
    strikes, time, call_price_surface = pmc.compute_call_prices(vol_params,
                                                                select_times,
                                                                True)
    with tf.Session() as session:
        res = session.run(call_price_surface)
    # plot results
    plot_surface(strikes, time, np.transpose(res))
    print("Close the plot window to continue...")
    plt.show()


def create_params_tensor(vary_correlation):
    """
    Create the model parameters tensor.

    Args:
        vary_correlation: If True correlation between variance and the process is a parameter.

    Returns:
        The parameters tensor.
    """

    lmbda = tf.Variable(np.float32(1.32))
    nu_bar = tf.Variable(np.float32(0.035))
    eta = tf.Variable(np.float32(0.38))

    clip_lmbda = tf.clip_by_value(lmbda, 0.0, 3.0)
    clip_nu_bar = tf.clip_by_value(nu_bar, 0.0, 1.0)
    clip_eta = tf.clip_by_value(eta, 0.0, 1.0)

    if vary_correlation:
        rho = tf.Variable(np.float32(0.0))
        clip_rho = tf.clip_by_value(rho, -1.0, 1.0)
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta, clip_rho])
    else:
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta])

    return vol_params


def setup_calibration_problem(vary_correlation):
    """
    Setup the calibration problem.

    Args:
        vary_correlation:  If True correlation between variance and the process is a parameter.

    Returns: Pair of tensors, the params and the loss .

    """
    # benchmarks
    selected_strikes, tenors, call_prices_by_fft, local_vol_spline_surface \
        = setup_local_vol('Heston', dict())
    # create the Monte Carlo
    pmc = HestonParticleCalibrator(local_vol_spline_surface, np.array(selected_strikes, np.float32))

    vol_params = create_params_tensor(vary_correlation)

    def select_times(time):
        """
        Select the maturity at 5.0 years.
        """
        return time == 5.0

    _, _, call_price_surface = pmc.compute_call_prices(vol_params,
                                                       select_times,
                                                       vary_correlation)

    lst = [tf.constant(row)
           for tenor, row in zip(tenors, call_prices_by_fft) if select_times(tenor)]

    temp = tf.reshape(tf.stack(lst) - call_price_surface, [-1])
    loss = tf.reduce_sum(tf.square(temp))
    return vol_params, loss


def test_gradients():
    """
    Compute the gradients with respect to model parameters.
    """
    vol_params, loss = setup_calibration_problem(vary_correlation=False)
    grads = tf.gradients(loss, vol_params)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
                            device_count={'GPU': 1})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        res = session.run([vol_params, loss, grads])
        print("vol_params:")
        print(res[0])
        print("loss:")
        print(res[1])
        print("grads:")
        print(res[2])


def calibrate():
    """
    Calibrate the model using stochastic gradient descent.
    """
    vol_params, loss = setup_calibration_problem(vary_correlation=False)
    loss = tf.Print(loss, [loss, vol_params], "loss, params :")
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # run the session
    #saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(train_step)
        # Save the variables to disk.
        #save_path = saver.save(session, "model.ckpt")
        #print "Model saved in file: %s" % save_path
        res = session.run(vol_params)
        print("Variables:")
        print(res)


def main():
    """ The main entry point function.
    """
    #test_surface()
    #test_gradients()
    parser = argparse.ArgumentParser()
    # parameters are a dictionary.
    parser.add_argument("-m", "--mode",
                        help="The mode.",
                        choices=["test_surface",
                                 "test_gradients",
                                 "calibrate"],
                        default="test_surface")
    # parse args
    args = parser.parse_args()
    if args.mode == "test_surface":
        test_surface()
    elif args.mode == "test_gradients":
        test_gradients()
    elif args.mode == "calibrate":
        calibrate()
    else:
        raise ValueError("Unknown mode.")

if __name__ == "__main__":
    main()
