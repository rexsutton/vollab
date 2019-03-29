
import numpy as np
import tensorflow as tf


class FlowHestonMonteCarlo(object):
    # spot = 50.0, short_rate = 0.05, dividend_yield = 0.02
    # sigma=0.3, lmbda=1.2, nu_bar=0.08, eta=1.8, rho=-0.45
    # sigma=0.3, lmbda=eta, nu_bar=nu_bar, eta=lmbda, rho=rho

    # "nu":0.0174, "nu_bar":0.0354, "eta":0.3877, "lmbda":1.3253, "rho":-0.7165

    def __init__(self, step_size = 0.1, num_steps = 50, num_paths=(4*2048), spot=50.0, drift=0.0, nu=0.0174, lmbda=1.3253, nu_bar=0.0354, eta=0.3877, rho=-0.7165):
        """
            Member-wise initialisation.
        """
        self.spot = np.float32(spot)
        self.drift = np.float32(drift)
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
        self.norms = None
        self.simulated_stock = None
        self.simulated_variance = None
        self.half = np.float32(0.5)
        self.quarter = np.float32(0.25)


    def log_stock_step(self, x, nu, norm, dt):
        return x + (self.drift - self.half * nu) * dt + tf.sqrt(nu * dt) * norm

    def variance_step(self, nu, norm, dt):

        temp1 = tf.sqrt(nu) + self.half * self.eta * np.sqrt(dt) * norm
        temp2 = -self.lmbda* (nu - self.nu_bar) * dt
        temp3 = -self.quarter * self.eta * self.eta * dt
        nu_new = temp1*temp1 + temp2 + temp3
        neg_nu_new = -1.0*nu_new
        return tf.maximum(nu_new, neg_nu_new)

    def build_sampler(self):

        mean_tensor = tf.constant([0.0,0.0])
        correl_tensor = tf.constant([[1.0, self.rho],[self.rho,1.0]])
        dist = tf.contrib.distributions.MultivariateNormalFullCovariance(mean_tensor, correl_tensor)

        initial_log_stock_values = tf.constant(np.full([self.num_paths], np.log(self.spot)))
        initial_variance_values = tf.constant(np.full([self.num_paths], self.nu))

        head = initial_log_stock_values, initial_variance_values, tf.exp(initial_log_stock_values)

        stack = []
        stack.append(head[2])
        for idx_step in range(self.num_steps):
            correl_rands_tensor = dist.sample([self.num_paths])
            norms1 = correl_rands_tensor[:,0]
            norms2 = correl_rands_tensor[:,1]
            log_stock_values = self.log_stock_step(head[0], head[1], norms1, self.step_size)
            variance_values = self.variance_step(head[1], norms2, self.step_size)
            head = log_stock_values, variance_values, tf.exp(log_stock_values)
            stack.append(head[2])

        stacker = tf.stack(stack)
        return stacker


def test():
    mc = FlowHestonMonteCarlo(num_steps=10, num_paths=8)
    session = tf.Session()
    res = session.run(mc.build_sampler())
    print(res)


def main():
    test()

if __name__ == "__main__":
    main()
