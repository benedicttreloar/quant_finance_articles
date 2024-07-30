import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

# Market Structure Defaults
DEFAULT_MARKET_SIZE = 500
DEFAULT_ESTIMATION_WINDOW = 252


def get_security_ids(market_size: int):
    return [f'sec_{x}' for x in range(market_size)]


class SimulatedMarket:

    def __init__(self,
                 factor_structure: dict,
                 stock_specific_vol: float,
                 market_size: int = DEFAULT_MARKET_SIZE,
                 security_ids:list = None,
                 estimation_window: int = DEFAULT_ESTIMATION_WINDOW):

        #### Generate a simulated market

        # This takes a set of market structure characteristics and generates a simulated market with those characteristics
        # Store factor structure and model characteristics
        self.__factor_structure = factor_structure
        self.__specific_vol = stock_specific_vol
        self.__estimation_window = estimation_window

        # Store market size and security identifier characteristics
        self._set_security_ids(market_size, security_ids)

        # Set market attributes for simulation
        self.reset_simulation()

    ### Ex-ante factor structure properties

    @property
    def factor_structure(self):
        return self.__factor_structure

    @property
    def factor_list(self):
        return list(self.__factor_structure.keys())

    @property
    def factor_vols(self):
        return {factor: params[1] for factor, params in self.factor_structure.items()}

    @property
    def specific_vol(self):
        return self.__specific_vol

    def _set_security_ids(self, market_size, security_ids):

        if security_ids is None:
            self.__security_ids = get_security_ids(market_size)
            self.__market_size = market_size
        else:
            self.__security_ids = security_ids
            self.__market_size = len(security_ids)

    @property
    def security_ids(self):
        return self.__security_ids

    @property
    def market_size(self):
        return self.__market_size

    @property
    def estimation_window(self):
        return self.__estimation_window

    @property
    def market_size(self):
        return self.__market_size

    @property
    def estimation_window(self):
        return self.__estimation_window

    ### Simulation methods
    def _get_exposures(self):

        ### Generate exposures for a market with the specified characteristics

        exposures = pd.DataFrame(index=self.security_ids, columns=self.factor_list)
        for factor in self.factor_structure.keys():
            if self.factor_structure[factor][2] == 'z-score':
                # For z-scored factors we draw exposures from a normal distribution
                exposures[factor] = np.random.normal(0, 1, self.market_size)
            elif self.factor_structure[factor][2] == 'dummy':
                # For dummy variables we use a Bernoulli distribution with the specified density
                exposures[factor] = np.where(
                    np.random.uniform(0, 1, self.market_size) <= self.factor_structure[factor][3], 1, 0)
            else:
                raise ValueError

        return exposures

    def _get_factor_returns(self):
        #### Generate factor returns for the desired time period
        factor_returns = pd.DataFrame(index=self.factor_list, columns=range(self.estimation_window))
        for factor in self.factor_structure.keys():
            mean = self.factor_structure[factor][0] / 252.
            vol = self.factor_structure[factor][1] / (252. ** 0.5)
            if vol > 0:
                # If the factor has volatility, draw from an appropriate normal distribution
                factor_returns.loc[factor] = np.random.normal(mean, vol, self.estimation_window)
            else:
                factor_returns.loc[factor] = pd.Series(mean, index=factor_returns.columns)

        return factor_returns

    def _get_idiosyncratic_returns(self):

        #### Generate idiosyncratic returns across stocks for the given time period
        idio_returns = pd.DataFrame(
            np.random.normal(0, self.specific_vol / (252 ** 0.5), (self.market_size, self.estimation_window)),
            index=self.security_ids, columns=range(self.estimation_window)
        )

        return idio_returns

    def _get_returns(self):

        #### Generate an array of security returns within the market

        # Generate true factor and idiosyncratic returns for the desired period
        factor_returns = self._get_factor_returns()
        idio_returns = self._get_idiosyncratic_returns()

        # Store the factor and idiosyncratic returns for future reference on the first run
        if self.__factor_returns is None:
            self.__factor_returns = factor_returns
        if self.__idio_returns is None:
            self.__idio_returns = idio_returns

        # Calculate security returns
        returns_out = self.exposures.dot(factor_returns) + idio_returns
        if self.__in_sample_returns is None:
            self.__in_sample_returns = returns_out.astype(float).copy()

        return returns_out.astype(float)

    ### Simulated Properties

    def exposures(self, date):

        '''Return the exposure dataframe'''
        # If exposures don't exist, create new exposures
        if self.__exposures is None:
            self.__exposures = self._get_exposures()
        if type(self.__exposures) == dict:
            return self.__exposures[date]
        else:
            return self.__exposures

    @property
    def factor_returns(self):

        if self.__factor_returns is None:
            self._get_returns()
        return self.__factor_returns

    @property
    def idio_returns(self):

        if self.__idio_returns is None:
            self._get_returns()
        return self.__idio_returns

    @property
    def in_sample_returns(self):

        if self.__in_sample_returns is None:
            self._get_returns()
        return self.__in_sample_returns

    @property
    def out_of_sample_returns(self):

        if self.__in_sample_returns is None:
            self._get_returns()
        if self.__out_of_sample_returns is None:
            self.__out_of_sample_returns = self._get_returns()
        return self.__out_of_sample_returns

    ### Simulation reset
    def reset_simulation(self):
        self.__exposures = None
        self.__factor_returns = None
        self.__idio_returns = None
        self.__in_sample_returns = None
        self.__out_of_sample_returns = None


class TrueModel:

    def __init__(self, input):

        """A model object which stores and returns model information for the true simulated market"""

        if type(input) == SimulatedMarket:
            self.__market = input
        else:
            raise ValueError('Input must be a SimulatedMarket object')

    @property
    def market(self):
        return self.__market

    @property
    def model_name(self):
        return 'true_model'

    @property
    def factor_structure(self):
        return self.market.factor_structure

    @property
    def factor_returns(self):
        return self.market.factor_returns

    @property
    def idio_returns(self):
        return self.market.idio_returns

    @property
    def security_ids(self):
        return self.market.security_ids

    @property
    def factor_list(self):
        return self.market.factor_list

    @property
    def factor_vols(self):
        return self.market.factor_vols

    @property
    def factor_mean(self):
        return pd.DataFrame(
            {'true_mean': {factor: params[0] for factor, params in self.market.factor_list}})

    @property
    def factor_covar(self):
        return pd.DataFrame({factor1:
            {
                factor2: vol ** 2 if factor1 == factor2 else 0.
                for factor2, vol in self.market.factor_vols.items()
            }
            for factor1 in self.market.factor_list
        })

    @property
    def idio_vols(self):
        return pd.Series(self.market.specific_vol, index=self.market.idio_returns.index, name='true_idio_vol')

    @property
    def rsq(self):
        return 0  # self.__rsq

    @property
    def tstat(self):
        return 0

