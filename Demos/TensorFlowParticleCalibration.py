
import argparse
import env
import json
import functools
import numpy as np
import tensorflow as tf
import TensorFlowCubicSpline as tfcs
import matplotlib.pyplot as plt
import vollab as vl

from mpl_toolkits.mplot3d import Axes3D

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
    z = np.zeros([len(x_axis), len(y_axis)], dtype=np.float64)

    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            z[i,j] = np.floor(z_values[i][j] / tol)*tol

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z)


class HestonParticleCalibrator(object):
    # spot = 50.0, short_rate = 0.05, dividend_yield = 0.02
    # sigma=0.3, lmbda=1.2, nu_bar=0.08, eta=1.8, rho=-0.45
    # sigma=0.3, lmbda=eta, nu_bar=nu_bar, eta=lmbda, rho=rho

    # "nu":0.0174, "nu_bar":0.0354, "eta":0.3877, "lmbda":1.3253, "rho":-0.7165

    def __init__(self, local_vol_surface, strikes, step_size = 0.5, num_steps = 10, num_paths=4*2048, spot=50.0, drift=0.0, nu=0.0174, lmbda=1.3253, nu_bar=0.0354, eta=0.3877, rho=-0.7165):
        """
            Member-wise initialisation.
        """
        self.local_vol_surface = local_vol_surface
        self.strikes = strikes
        self.spot = np.float32(spot)
        self.nu = np.float32(nu)
        self.lmbda = np.float32(lmbda)
        self.nu_bar = np.float32(nu_bar)
        self.eta = np.float32(eta)
        self.rho = np.float32(rho)
        self.step_size = np.float32(step_size)
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.time = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.half = np.float32(0.5)
        self.quarter = np.float32(0.25)
        self.sigma_variance_swap = np.float32(0.3)
        self.kappa = np.float32(1.5)
        self.time_floor = np.float32(0.25)

        self.sqrt_step_size = np.float32(np.sqrt(step_size))
        self.initial_local_vol = np.float32(self.local_vol_surface.__call__(self.spot, 0.0))

        self.num_strikes = len(self.strikes)
        self.r_tensor = tf.constant(np.float32(15.0 / 16.0))
        self.one_tensor = tf.constant(np.float32(1.0))
        self.ones_tensor = tf.constant(np.full([self.num_paths], np.float32(1.0)))
        self.zero_tensor = tf.constant(np.float32(0.0))
        self.zeros_tensor = tf.constant(np.full([self.num_paths], np.float32(0.0)))
        self.num_paths_tensor = tf.constant(np.float32(int(self.num_paths)))
        self.mean_tensor = tf.constant([0.0, 0.0])
        self.zeros_paths_tensor = tf.constant(np.full([self.num_paths], np.float32(0.0)))

        self.strikes_tensor = tf.constant(np.array(strikes, np.float32))
        self.strike_tensors = []
        for strike in strikes:
            self.strike_tensors.append(tf.constant(np.full([num_paths], np.float32(strike))))

        self.strike_min_tensor = tf.constant(np.full([num_paths], np.min(strikes), np.float32))
        self.strike_max_tensor = tf.constant(np.full([num_paths], np.max(strikes), np.float32))

        self.log_strike_min_tensor = tf.constant(np.full([num_paths], np.log(np.min(strikes)), np.float32))
        self.log_strike_max_tensor = tf.constant(np.full([num_paths], np.log(np.max(strikes)), np.float32))
        # initial values
        self.initial_log_forward_tensor = tf.constant(np.full([num_paths], np.log(self.spot)))
        self.initial_forward_tensor = tf.constant(np.full([num_paths], self.spot))
        self.initial_effective_volatility_tensor = tf.constant(np.full([num_paths], self.initial_local_vol / np.sqrt(self.nu)))
        self.initial_stochastic_variance_tensor = tf.constant(np.full([num_paths], self.nu))


    def effective_volatility(self,
                             current_forward,
                             maturity_time_in_years,
                             particle_forwards,
                             particle_stochastic_variance):

        """
        Calculate the effective volatility for the forward points.

        Args:
            local_vol_surface(spline surface): The local volatility surface.
            forward_points (vector): The points __call__ which effective volatility is calculated.
            particle_forwards (vector): The particle forwards.
            particle_variance (vector): The particle volatility.

        Returns: The effective volatility calculated for the forward points.

        """
        h = self.kappa * current_forward * self.sigma_variance_swap * np.sqrt(maturity_time_in_years if maturity_time_in_years > self.time_floor else self.time_floor) / (self.num_paths ** 0.2)
        h_tensor = tf.constant(np.float32(h))
        sigma = tf.constant(np.array([self.local_vol_surface(strike, maturity_time_in_years) for strike in self.strikes], np.float32))

        contribs = []
        for strike_tensor in self.strike_tensors:
            difs =  (particle_forwards - strike_tensor) / h_tensor
            cond = tf.less(tf.abs(difs), self.ones_tensor)
            temp1 = self.ones_tensor - difs * difs
            delta_array_values = self.r_tensor * (temp1 * temp1)
            delta_array = tf.where(cond, delta_array_values, self.zeros_tensor) / h_tensor
            delta_sum = tf.reduce_sum(delta_array)
            variance_sum = tf.reduce_sum(delta_array * particle_stochastic_variance)
            contribs.append(tf.sqrt(delta_sum / variance_sum))

        temp = tf.stack(contribs)
        contrib = tf.reshape(temp, [int(len(self.strikes))])

        #return tfcs.cubic_spline(self.strikes, sigma * contrib, cut_forwards), condition

        # cut-off particle forwards beyond the max strike and below min_strike
        condition1 = tf.less(particle_forwards, self.strike_max_tensor)
        condition2 = tf.greater(particle_forwards, self.strike_min_tensor)
        condition = tf.logical_and(condition1, condition2)
        cut_forwards = tf.where(condition, particle_forwards, self.initial_forward_tensor)
        spline_values = tfcs.cubic_spline(self.strikes, sigma * contrib, cut_forwards)
        return tf.where(condition, spline_values, self.ones_tensor)


    def call_prices(self, forwards_tensor):
        lst = []
        for strike_tensor in self.strike_tensors:
            difs = forwards_tensor - strike_tensor
            cond = tf.greater(difs, self.zeros_tensor)
            intrinsic = tf.where(cond, difs, self.zeros_tensor)
            call_price = tf.reduce_sum(intrinsic) / self.num_paths_tensor
            lst.append(call_price)
        return tf.stack(lst)

    # nu=0.0174,
    # param 0 lmbda=1.3253, param 1 nu_bar=0.0354, param 2 eta=0.3877, param 3 rho=-0.7165

    def variance_step(self, nu, norm, vol_params, dt):

        temp1 = tf.sqrt(nu) + self.half * vol_params[2] * np.sqrt(dt) * norm
        temp2 = -vol_params[0] * (nu - vol_params[1]) * dt
        temp3 = -self.quarter * vol_params[2] * vol_params[2] * dt
        nu_new = temp1 * temp1 + temp2 + temp3
        neg_nu_new = -1.0 * nu_new
        return tf.maximum(nu_new, neg_nu_new)

    def calculate_call_prices_particle_method(self, vol_params, select_times, vary_correlation):
        # sampler for brownians, with parameterised correlation
        if vary_correlation:
            corr_tensor = vol_params[3]
        else:
            corr_tensor = tf.constant(np.float32(-0.7165))

        correl_tensor = tf.stack([tf.stack([self.one_tensor, corr_tensor]), tf.stack([corr_tensor, self.one_tensor])])
        dist = tf.contrib.distributions.MultivariateNormalFullCovariance(self.mean_tensor, correl_tensor)
        head = self.initial_log_forward_tensor, self.initial_forward_tensor, self.initial_effective_volatility_tensor,  self.initial_stochastic_variance_tensor
        # run simulation
        call_price_surface = []
        if(select_times(0.0)):
            print 'selected:0.0'
            call_price_surface.append(self.call_prices(head[1]))
        # for each times (not including zero)
        for time in self.time:
            # sample brownians
            correl_rands_tensor = dist.sample([self.num_paths])
            norms1 = correl_rands_tensor[:, 0]
            norms2 = correl_rands_tensor[:, 1]
            # update the forwards
            stochastic_vol = tf.sqrt(head[3])
            #ORIG
            #temp = head[0] + head[2] * stochastic_vol * self.sqrt_step_size * norms1
            # cut-off forwards beyond the max strike and below min_strike
            #condition1 = tf.less(temp, self.log_strike_max_tensor)
            #condition2 = tf.greater(temp, self.log_strike_min_tensor)
            #condition = tf.logical_and(condition1, condition2)
            #log_forward = tf.where(condition, temp, self.initial_log_forward_tensor)
            sigma = head[2] * stochastic_vol
            log_forward = head[0] - 0.5 * sigma * sigma * self.step_size + sigma * self.sqrt_step_size * norms1
            forwards = tf.exp(log_forward)
            stochastic_variance = self.variance_step(head[3], norms2, vol_params, self.step_size)
            effective_vol = self.effective_volatility(self.spot, time, forwards, stochastic_variance)
            head = log_forward, forwards, effective_vol, stochastic_variance
            # update call prices
            if(select_times(time)):
                print 'selected:', time
                call_price_surface.append(self.call_prices(head[1]))
        # return
        return self.strikes, self.time_axis, tf.stack(call_price_surface)


def test_surface():
    # create market parameters
    market_params = vl.MarketParams()
    # market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function('Heston')
    # characteristic_function.__dict__.update(params)
    # select the range of strikes to plot
    strike_selector = functools.partial(vl.select_strike, 0.7 * market_params.spot, 1.3 * market_params.spot)
    # select local vol tenors and simulation times
    tenors = np.linspace(0.0, 50 * 0.1, 1 + 50)
    floored_tenors = np.array([tenor for tenor in tenors if tenor > 0.25])
    print floored_tenors
    # calculate call prices benchmark
    selected_strikes, tenors, call_prices_by_tenor_by_strike = vl.compute_call_prices_matrix(characteristic_function,
                                                                                             market_params,
                                                                                             strike_selector,
                                                                                             tenors)
    # calculate the local vol matrix
    selected_strikes, floored_tenors, local_vol_matrix = vl.compute_local_vol_matrix(characteristic_function,
                                                                                     market_params,
                                                                                     strike_selector,
                                                                                     floored_tenors)
    print len(floored_tenors), len(selected_strikes)
    local_vol_spline_surface = vl.SplineSurface(selected_strikes, floored_tenors, local_vol_matrix, tenors)
    # create the calibrator
    mc = HestonParticleCalibrator(local_vol_spline_surface, np.array(selected_strikes, np.float32))
    vol_params = tf.constant(np.array([1.3253, 0.0354, 0.3877, -0.7165], np.float32))
    # use particle method to calculate call prices
    select_times = lambda x: True
    strikes, time, call_price_surface = mc.calculate_call_prices_particle_method(vol_params, select_times, True)
    session = tf.Session()
    res = session.run(call_price_surface)
    print res
    print len(strikes), len(time), len(res), len(res[0])
    plot_surface(selected_strikes, tenors, np.transpose(call_prices_by_tenor_by_strike))
    plot_surface(strikes, time, np.transpose(res))
    print "Close the plot window to continue..."
    plt.show()


def test0():
    vary_correlation = False
    # create market parameters
    market_params = vl.MarketParams()
    # market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function('Heston')
    # characteristic_function.__dict__.update(params)
    # select the range of strikes to plot
    strike_selector = functools.partial(vl.select_strike, 0.7 * market_params.spot, 1.3 * market_params.spot)
    # select local vol tenors and simulation times
    tenors = np.linspace(0.0, 50 * 0.1, 1 + 50)
    floored_tenors = np.array([tenor for tenor in tenors if tenor > 0.25])
    # calculate the local vol matrix
    selected_strikes, floored_tenors, local_vol_matrix = vl.compute_local_vol_matrix(characteristic_function,
                                                                                     market_params,
                                                                                     strike_selector,
                                                                                     floored_tenors)
    local_vol_spline_surface = vl.SplineSurface(selected_strikes, floored_tenors, local_vol_matrix, tenors)
    # create the calibrator
    mc = HestonParticleCalibrator(local_vol_spline_surface, np.array(selected_strikes, np.float32))
    # calculate call prices benchmark
    selected_strikes, tenors, call_prices_by_tenor_by_strike = vl.compute_call_prices_matrix(characteristic_function,
                                                                                             market_params,
                                                                                             strike_selector,
                                                                                             mc.time_axis)

    # param 0 lmbda=1.3253, param 1 nu_bar=0.0354, param 2 eta=0.3877, param 3 rho=-0.7165
    #lmbda = tf.Variable(np.float32(0.0))
    #nu_bar = tf.Variable(np.float32(0.0))
    #eta = tf.Variable(np.float32(1.0))
    #rho = tf.Variable(np.float32(0.0))

    lmbda = tf.Variable(np.float32(1.32))
    nu_bar = tf.Variable(np.float32(0.035))
    eta = tf.Variable(np.float32(0.38))
    rho = tf.Variable(np.float32(-0.71))

    clip_lmbda = tf.clip_by_value(lmbda, 0.0, 3.0)
    clip_nu_bar = tf.clip_by_value(nu_bar, 0.0, 1.0)
    clip_eta = tf.clip_by_value(eta, 0.0, 1.0)

    if vary_correlation:
        rho = tf.Variable(np.float32(0.0))
        clip_rho = tf.clip_by_value(rho, -1.0, 1.0)
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta, clip_rho])
    else:
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta])

    select_times = lambda x: x==5.0
    strikes, time, call_price_surface = mc.calculate_call_prices_particle_method(vol_params, select_times, vary_correlation)

    lst = []
    for tenor, row in zip(tenors, call_prices_by_tenor_by_strike):
        if select_times(tenor):
            lst.append(tf.constant(row))
    mtx = tf.stack(lst)

    # use particle method to calculate call prices
    temp = tf.reshape(mtx - call_price_surface, [-1])
    loss = tf.reduce_sum(temp * temp)

    grads = tf.gradients(loss, vol_params)

    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), device_count = {'GPU': 1})

    with tf.Session(config=config) as session:
        session.run(init_op)
        res = session.run([vol_params, loss, grads])
        print "vol_params:"
        print res[0]
        print "loss:"
        print res[1]
        print "grads:"
        print res[2]


def test():
    vary_correlation = False
    # create market parameters
    market_params = vl.MarketParams()
    # market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_charac                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          teristic_function('Heston')
    # characteristic_function.__dict__.update(params)
    # select the range of strikes to plot
    strike_selector = functools.partial(vl.select_strike, 0.7 * market_params.spot, 1.3 * market_params.spot)
    # select local vol tenors and simulation times
    tenors = np.linspace(0.0, 50 * 0.1, 1 + 50)
    floored_tenors = np.array([tenor for tenor in tenors if tenor > 0.25])
    # calculate the local vol matrix
    selected_strikes, floored_tenors, local_vol_matrix = vl.compute_local_vol_matrix(characteristic_function,
                                                                                     market_params,
                                                                                     strike_selector,
                                                                                     floored_tenors)
    local_vol_spline_surface = vl.SplineSurface(selected_strikes, floored_tenors, local_vol_matrix, tenors)
    # create the calibrator
    mc = HestonParticleCalibrator(local_vol_spline_surface, np.array(selected_strikes, np.float32))
    # calculate call prices benchmark
    selected_strikes, tenors, call_prices_by_tenor_by_strike = vl.compute_call_prices_matrix(characteristic_function,
                                                                                             market_params,
                                                                                             strike_selector,
                                                                                             mc.time_axis)

    # param 0 lmbda=1.3253, param 1 nu_bar=0.0354, param 2 eta=0.3877, param 3 rho=-0.7165
    lmbda = tf.Variable(np.float32(0.0))
    nu_bar = tf.Variable(np.float32(0.0))
    eta = tf.Variable(np.float32(1.0))

    clip_lmbda = tf.clip_by_value(lmbda, 0.0, 3.0)
    clip_nu_bar = tf.clip_by_value(nu_bar, 0.0, 1.0)
    clip_eta = tf.clip_by_value(eta, 0.0, 1.0)

    if vary_correlation:
        rho = tf.Variable(np.float32(0.0))
        clip_rho = tf.clip_by_value(rho, -1.0, 1.0)
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta, clip_rho])
    else:
        vol_params = tf.stack([clip_lmbda, clip_nu_bar, clip_eta])

    select_times = lambda x: x==5.0

    strikes, time, call_price_surface = mc.calculate_call_prices_particle_method(vol_params, select_times, vary_correlation)

    lst = []
    for tenor, row in zip(tenors, call_prices_by_tenor_by_strike):
        if select_times(tenor):
            lst.append(tf.constant(row))
    mtx = tf.stack(lst)

    # use particle method to calculate call prices
    temp = tf.reshape(mtx - call_price_surface, [-1])
    loss = tf.reduce_sum(tf.square(temp))
    loss = tf.Print(loss, [loss, vol_params], "loss, params")

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()

    #config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), device_count = {'GPU': 1})
    # run the session
    #with tf.Session(config=config) as session:
    with tf.Session() as session:
    #tf.global_variables_initializer().run()
        session.run(init_op)
        session.run(train_step)
        # Save the variables to disk.
        save_path = saver.save(session, "model.ckpt")
        print("Model saved in file: %s" % save_path)
        res = session.run(vol_params)
        print "Variables:"
        print res
    #print res
    #print len(strikes), len(times), len(res), len(res[0])
    #plot_surface(selected_strikes, tenors, np.transpose(call_prices_by_tenor_by_strike))
    #plot_surface(strikes, times, np.transpose(res))
    #print "Close the plot window to continue..."
    #plt.show()


def main():
    #test_surface()
    test0                                                                                                                                                                                                                                                                                           ()

if __name__ == "__main__":
    main()
