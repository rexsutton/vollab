"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Calculate the prices of european call options,
     for stochastic processes with known characteristic function using Fast Fourier Transform.

    NOTE As this is demo code,
     the spot price must be in the range of 100.0 and no re-scaling has been implemented.

    The code is based on the paper of Carr-Madan:-

    @article{carr1999option,
      title={Option valuation using the fast Fourier transform},
      author={Carr, Peter and Madan, Dilip},
      journal={Journal of computational finance},
      volume={2},
      number={4},
      pages={61--73},
      year={1999}
    }
"""

import numpy as np


class MarketParams(object):  # pylint: disable=too-few-public-methods
    """
        Encapsulate market parameters in a structure.
    """
    def __init__(self, spot=50.0, short_rate=0.0, dividend_yield=0.0):
        """
            Member-wise initialisation.
        Args:
            spot: The current value of the underlying process.
            short_rate: The rate of interest.
            dividend_yield: The dividend yield.
        """
        self.spot = spot
        self.short_rate = short_rate
        self.dividend_yield = dividend_yield

    def forward(self, maturity_time_years):
        """
            Calculate the price agreed today to receive the underlying at the maturity time,
             under no-arbitrage assumptions.
        Args:
            maturity_time_years: The times of exchange, measured in years relative to today,
             today being zero.

        Returns:
            The forward price.

        """
        return self.spot * np.exp((self.short_rate - self.dividend_yield) * maturity_time_years)


class BlackScholesCharacteristicFunction(object):  # pylint: disable=too-few-public-methods
    """
        The characteristic function of the underyling under the Black-Scholes model.

    """
    def __init__(self, sigma=0.2):
        """
            Simple initialisation.
        Args:
            sigma: The volatility of the underlying.
        """
        self.sigma = sigma

    def __call__(self, market_params, maturity_time_years, variable):
        """
            Calculate the characteristic function of the underyling.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            variable: The function argument.

        Returns:
            The value of the characteristic function.

        """
        i = complex(0.0, 1.0)
        temp1 = (((market_params.short_rate - market_params.dividend_yield) * maturity_time_years)
                 + np.log(market_params.spot)) * i * variable
        temp2 = 0.5 * variable * (variable + i) * self.sigma * self.sigma * maturity_time_years
        temp3 = temp1 - temp2
        temp4 = np.exp(temp3)
        return temp4


class HestonCharacteristicFunction(object):  # pylint: disable=too-few-public-methods
    """
        The characteristic function of the underyling under the Heston stochastic volatility model.
    """
    def __init__(self, nu=0.0174, lmbda=1.3253, nu_bar=0.0354, eta=0.3877, rho=-0.7165): #pylint: disable=too-many-arguments
        """
            Simple initialisation.
        Args:
            nu: The initial variance.
            nu_bar: The long run variance.
            lmbda: The mean reversion rate.
            eta : The volatility of variance.
            rho: The correlation between volatility and the underlying.
        """
        self.nu = nu #pylint: disable=invalid-name
        self.nu_bar = nu_bar
        self.eta = eta
        self.lmbda = lmbda
        self.rho = rho

    def __call__(self, market_params, maturity_time_years, variable):
        """
            Calculate the characteristic function of the underyling.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            variable: The function argument.

        Returns:
            The value of the characteristic function.

        """
        i = complex(0.0, 1.0)

        temp0 = self.lmbda - (self.rho * self.eta * i * variable)
        d_variable = np.sqrt((temp0 * temp0)
                             + self.eta * self.eta * ((i * variable) + (variable * variable)))
        temp1 = temp0 - d_variable
        g2_variable = temp1 / (temp0 + d_variable)
        temp2 = np.exp(-1.0 * d_variable * maturity_time_years)
        temp3 = 1.0 - (g2_variable * temp2)
        temp4 = np.exp(i * variable * (
            np.log(market_params.spot) + (
                (market_params.short_rate - market_params.dividend_yield) * maturity_time_years)))
        temp5 = (self.nu_bar * self.lmbda / (self.eta * self.eta)) * (
            (temp1 * maturity_time_years) - (2.0 * np.log(temp3 / (1.0 - g2_variable))))
        temp6 = ((self.nu) / (self.eta * self.eta)) * temp1 * (1.0 - temp2) / temp3

        return temp4 * np.exp(temp5) * np.exp(temp6)


class VarianceGammaCharacteristicFunction(object): # pylint: disable=too-few-public-methods
    """
        The characteristic function of the underyling under the Variance-Gamma model.

    """
    def __init__(self, theta=-0.14, nu=0.2, sigma=0.12):
        """
            Member-wise initialisation.
        Args:
            theta:
            nu:
            sigma:
        """
        self.theta = theta
        self.nu = nu  #pylint: disable=invalid-name
        self.sigma = sigma
        self.omega = np.log(1.0 - (theta * nu) - (sigma * sigma * nu * 0.5)) / nu

    def __call__(self, market_params, maturity_time_years, variable):
        """
            Calculate the characteristic function of the underyling.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            variable: The function argument.

        Returns:
            The value of the characteristic function.

        """
        i = complex(0.0, 1.0)
        tmp0 = np.log(market_params.spot) + (
            maturity_time_years * (market_params.short_rate
                                   - market_params.dividend_yield
                                   + self.omega))
        tmp1 = np.exp(i * variable * tmp0)
        tmp2 = 1.0 - (i * self.theta * self.nu * variable)\
               + (0.5 * self.sigma * self.sigma * self.nu * variable * variable)
        tmp3 = np.power(tmp2, (maturity_time_years / self.nu))
        tmp4 = tmp1 / tmp3
        return tmp4


def compute_psi(alpha, market_params, maturity_time_years, characteristic_function, variable):
    """
    Calculate characteristic function dependent quantity in the integrand term.
    Args:
        alpha: Regularization parameter.
        market_params: The market parameters.
        maturity_time_years: Measured in years relative to today, today being zero.
        characteristic_function: The characteristic function of the density.
        variable: The argument.

    Returns:
        Intermediate quantity in the integrand term in the Fourier transform.

    """
    temp1 = complex(variable, -1.0 * (alpha + 1.0))
    temp2 = characteristic_function(market_params, maturity_time_years, temp1)
    temp3 = 1.0 / complex((alpha * (alpha + 1.0)) - (variable * variable),
                          ((2.0 * alpha) + 1.0) * variable)
    temp4 = temp2 * temp3
    return temp4


def compute_weight(i):
    """
        Compute weighting according to Simpsons's rule of i'th integrand term.
    Args:
        i: Integer, the index of the term in the integrand.

    Returns:
        Weighting according to Simpsons's rule of i'th integrand term.

    """
    delta = 1.0 if i == 1 else 0.0
    power = 1.0 if i % 2 == 0 else -1.0
    return (3.0 + power - delta) / 3.0


class CallPriceCalculator(object):
    """
        Function object for calculating European call option prices as a function of strike.
    """
    def __init__(self, num_points, lmbda, alpha):
        """

        Args:
            num_points: The number of points on the strike axis to consider, power of two.
            lmbda: Separation in the log-strike axis.
            alpha: Regularization parameter.
        """
        # fidelity in the transform axis,
        #  cannot be independently varied with respect to the log-strike axis
        self.alpha = alpha
        self.lmbda = lmbda
        self.eta = (2.0 * np.pi) / (self.lmbda * num_points)
        self.offset = num_points / 2
        self.transform_axis = [i * self.eta for i in range(0, num_points)]
        self.log_strike_axis = [(i - self.offset) * self.lmbda for i in range(0, num_points)]
        self.strike_axis = [np.exp(log_strike) for log_strike in self.log_strike_axis]

    def compute_integrand_sequence(self,
                                   market_params,
                                   maturity_time_years,
                                   characteristic_function):
        """
            Compute the integrand sequence for the transform.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            characteristic_function: The characteristic function.

        Returns:
            The integrand sequence.
        """
        log_strike_lower_bound = -1.0 * self.log_strike_axis[0]
        cmplex = []
        for j, variable in enumerate(self.transform_axis):
            psi = compute_psi(self.alpha,
                              market_params,
                              maturity_time_years,
                              characteristic_function,
                              variable)
            temp5 = np.exp(complex(0.0, log_strike_lower_bound * variable))
            weight = compute_weight(j + 1)
            cmplex.append(psi * temp5 * weight)

        return cmplex

    def apply_multipliers(self, market_params, maturity_time_years, complexs):
        """
            Apply multipliers to the result of the transform to retrieve the call prices.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            complexs: The results of the transform.

        Returns:
            A sequence of call-prices for strikes on the strikes_axis.
        """

        mul = np.exp(-1.0 * market_params.short_rate * maturity_time_years) * self.eta / np.pi

        call_prices = []
        for strike, cmplex in zip(self.strike_axis, complexs):
            # exp(-alpha k) = exp(-alpha ln K) = K^-alpha
            temp = mul * np.power(strike, - 1.0 * self.alpha) * cmplex
            call_prices.append(temp.real)
        return call_prices

    def compute_call_prices(self, market_params, maturity_time_years, characteristic_function):
        """
            Compute the sequence of call-prices,
             for strikes on the strikes axis for the specified maturity times.
        Args:
            market_params: The market parameters.
            maturity_time_years: Measured in years relative to today, today being zero.
            characteristic_function: The characteristic function of the model.

        Returns:
            The sequence of call-prices for strikes on the strikes_axis
        """
        integrand_sequence = self.compute_integrand_sequence(market_params,
                                                             maturity_time_years,
                                                             characteristic_function)
        values = np.fft.fft(integrand_sequence)
        return self.apply_multipliers(market_params, maturity_time_years, values)


def create_characteristic_function(name):
    """
        Factory for characteristic function.
    Args:
        name: The name of the model.
    Returns:
        The characteristic function with default parameters.

    """
    factory = {"BlackScholes": BlackScholesCharacteristicFunction,
               "Heston": HestonCharacteristicFunction,
               "VarianceGamma": VarianceGammaCharacteristicFunction}
    try:
        return factory[name]()
    except:
        raise Exception("Unknown model:" + name)


def select_strike(strike_lower, strike_upper, strike):
    """
        Predicate for the selection of the strike.
    Args:
        strike_lower: The lower bound.
        strike_upper: The upper bound.
        strike: The strike.

    Returns:
        True if the strike satisfies the predicate.
    """
    return strike_lower < strike < strike_upper


def calculate_prices_at_maturity(call_price_calculator,
                                 characteristic_function,
                                 market_params,
                                 maturity_time,
                                 strike_selector):
    """

    Calculate a list of call prices in order of strike, satisfying the strike selection criteria,
     for the given expiry tenor.

    Args:
        call_price_calculator: The call price calculator.
        characteristic_function: The characteristic function.
        market_params: The market parameters.
        maturity_time: The times the option expires.
        strike_selector: Predicate function for selecting strikes.

    Returns:
        A list of call prices in order of strike, for the given expiry tenor.

    """
    # calculate the call prices
    call_prices = call_price_calculator.compute_call_prices(market_params,
                                                            maturity_time,
                                                            characteristic_function)
    # select the values
    ret = []
    # for selected strikes
    for strike, call_price in zip(call_price_calculator.strike_axis, call_prices):
        if strike_selector(strike):
            ret.append(call_price)
    return ret


def compute_call_prices_matrix(characteristic_function,
                               market_params,
                               strike_selector,
                               maturity_times):
    """
    Calculate matrix of call prices, in increasing order of maturity,
     and increasing strike at each maturity time.

    Args:
        characteristic_function: The characteristic function.
        market_params:The market parameters.
        strike_selector: Predicate function for selecting strikes.
        maturity_times: The maturity tenors of interest.

    Returns:
        A matrix of call prices for the given strikes and tenors.

    """
    # create the call price calculator
    call_price_calculator = CallPriceCalculator(num_points=4096, lmbda=0.005, alpha=1.0)
    # select the range of strikes
    selected_strikes = [strike for strike in call_price_calculator.strike_axis
                        if strike_selector(strike)]
    # generate a surface for the following maturity times in years
    surface = []
    for tenor in maturity_times:
        # append smile to the surface
        surface.append(calculate_prices_at_maturity(call_price_calculator,
                                                    characteristic_function,
                                                    market_params,
                                                    tenor,
                                                    strike_selector))
    return selected_strikes, maturity_times, surface
