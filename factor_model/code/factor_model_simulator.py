import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

# Market Structure Defaults
DEFAULT_MARKET_SIZE = 500
DEFAULT_ESTIMATION_WINDOW = 252



def get_security_ids(market_size: int):
    return [f'sec_{x}' for x in range(market_size)]


class SimulatedMarket():

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

    @property
    def exposures(self):

        if self.__exposures is None:
            self.__exposures = self._get_exposures()
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
        return 0 #self.__rsq

    @property
    def tstat(self):
        return 0


class Model:

    def __init__(self, market: SimulatedMarket, model_name:str):

        self.__market = market
        self.__model_name = model_name

    @property
    def market(self):
        return self.__market

    @property
    def model_name(self):
        return self.__model_name

    @property
    def fit(self, excluded_factors: list = None, method='ols'):

        # Fit factor model
        if method == 'ols':
            self._fit_returns(excluded_factors)
        else:
            raise NotImplementedError

    def _fit_returns(self, excluded_factors: list = None):

        #### Fit a fundamental factor model of market returns

        # Store excluded factors
        self.__excluded_factors = excluded_factors

        # Generate a reduced form factor exposure matrix where applicable
        if excluded_factors is not None:
            excluded_factors_explicit = self._get_excluded_factor_list(excluded_factors)
            input_exposures = self.market.exposures.drop(excluded_factors_explicit, axis=1).copy()
        else:
            input_exposures = self.market.exposures.copy()
        input_returns = self.market.in_sample_returns.copy()

        # Estimate model
        model_factors_dict = {}
        model_idio_dict = {}
        model_rsq_dict = {}
        model_tstat_dict = {}

        # Period by period, estimate within-period factor returns and idiosyncratic returns to go with it
        for date in input_returns.columns:
            # Estimate factor and idiosyncratic returns
            mod = sm.OLS(input_returns[date], input_exposures)
            res = mod.fit()
            model_factors_dict[date] = res.params.transpose()
            model_idio_dict[date] = res.resid

            # Collect model power data
            model_rsq_dict[date] = res.rsquared
            model_tstat_dict[date] = res.tvalues

        # Generate output dataframes
        model_factors = pd.DataFrame(model_factors_dict)
        for factor in [x for x in self.market.factor_list if x not in model_factors.index]:
            model_factors.loc[factor] = 0.
        self.__factor_returns = model_factors.loc[self.market.factor_list].copy()
        self.__idio_returns = pd.DataFrame(model_idio_dict)

        # Generate output statistical information
        self.__rsq = pd.Series(model_rsq_dict)
        model_tstat = pd.DataFrame(model_tstat_dict)
        for factor in [x for x in self.market.factor_list if x not in model_tstat.index]:
            model_tstat.loc[factor] = 0.
        self.__tstat = model_tstat.loc[self.market.factor_list].copy()

    @property
    def factor_returns(self):
        return self.__factor_returns

    @property
    def idio_returns(self):
        return self.__idio_returns

    @property
    def rsq(self):
        return self.__rsq

    @property
    def tstat(self):
        return self.__tstat

    def _get_excluded_factor_list(self, excluded_factors: list):

        # Generate a list of columns to exclude from the return estimation piece
        if type(excluded_factors) == str:
            excluded_factors = [excluded_factors]

        # Exclude exact matches
        excluded_factors_explicit = [x for x in self.__market.factor_list if x in excluded_factors]
        if len(excluded_factors_explicit) > 0:
            return excluded_factors_explicit

        # Exclude via partial matching
        excluded_factors_explicit = []
        for factor in excluded_factors:
            excluded_factors_explicit += [x for x in self.__market.factor_list if factor in x]
        return list(set(excluded_factors_explicit))

    @property
    def factor_list(self):
        return list(self.__market.factor_list)

    @property
    def factor_vols(self):
        try:
            return (self.factor_returns.std(axis=1) * (252. ** 0.5)).to_dict()
        except:
            raise ValueError('Model factor returns have not been estimated')

    @property
    def factor_mean(self):
        try:
            return (self.factor_returns.mean(axis=1) * 252.).to_dict()
        except:
            raise ValueError('Model factor returns have not been estimated')

    @property
    def factor_covar(self):

        try:
            return self.factor_returns.transpose().cov() * 252
        except:
            raise ValueError('Model factor returns have not been estimated')

    @property
    def idio_vols(self):
        try:
            return self.idio_returns.std(axis=1).rename('model_idio_vol') * (252. ** 0.5)
        except:
            raise ValueError('Model idiosyncratic returns have not been estimated')


# Generate a set of random portfolios
class Portfolios:

    def __init__(self, market, port_size=DEFAULT_PORTFOLIO_SIZE, port_number=DEFAULT_PORTFOLIO_NUMBER, port_ids=None):
        if type(market) == int:
            self._generate_market(market)
        elif type(market) == SimulatedMarket:
            self.__market = market
        self.__port_size = port_size
        self._set_port_ids(port_number, port_ids)
        self._generate_weights()

    def _generate_market(self, market_size):
        self.__market_size = market_size
        self.__security_ids = get_security_ids(market_size)

    def _reset_simulation(self):
        self.__weights = pd.DataFrame(0., index=self.security_ids, columns=self.port_ids)

    def _generate_weights(self):
        self._reset_simulation()
        for port in tqdm(self.port_ids):
            self.__weights.loc[
                np.random.choice(self.security_ids, self.port_size, replace=False).tolist(), port] = 1 / self.port_size

    def _set_port_ids(self, port_number, port_ids=None):

        if port_ids is None:
            self.__port_number = port_number
            self.__port_ids = [f'port_{i}' for i in range(port_number)]
        else:
            self.__port_number = len(port_ids)
            self.__port_ids = port_ids

    @property
    def weights(self):
        return self.__weights

    @property
    def exposures(self):
        try:
            return self.weights.transpose().dot(self.market.exposures)
        except:
            try:
                self.weights.transpose()
                raise ValueError('Portfolio weights corrupted')
            except:
                raise NotImplementedError('Market exposures have not been not generated')

    @property
    def market(self):
        return self.__market

    @property
    def port_size(self):
        return self.__port_size

    @property
    def port_number(self):
        return self.__port_number

    @property
    def port_ids(self):
        return self.__port_ids

    @property
    def security_ids(self):
        try:
            return self.market.security_ids
        except:
            return self.__security_ids

    @property
    def market_size(self):
        try:
            return self.market.market_size
        except:
            return self.__market_size

    def returns(self, return_type:str):

        # Calculate portfolio level return timeseries
        if return_type == 'in_sample':
            return self.weights.transpose().dot(self.market.in_sample_returns)
        elif return_type == 'out_of_sample':
            return self.weights.transpose().dot(self.market.out_of_sample_returns)
        elif return_type == 'random':
            raise NotImplementedError('Random returns not implemented')
        else:
            raise NotImplementedError('Return type specified not implemented')

    def ex_post_vol(self, return_type: str):

        # Calculate empirical volatility of returns
        return (self.returns(return_type).std(axis=1) * (252 ** 0.5)).rename(f'{return_type}_ex_post_vol')

    def model_car(self, model):

        # Generate CAR breakdown for the input model
        port_var_contribs = self.exposures.dot(model.factor_covar) * self.exposures
        port_var_contribs['idio'] = (self.weights ** 2).transpose().dot(model.idio_vols ** 2)
        return (port_var_contribs.transpose() / (np.sum(port_var_contribs, axis=1) ** 0.5)).transpose()

    def model_risk(self, model):

        # Generate portfolio risk for the input model
        port_risk = ((np.diag(self.exposures.dot(model.factor_covar).dot(self.exposures.transpose()))
                     + (self.weights ** 2).transpose().dot(model.idio_vols ** 2)) ** 0.5
                     ).rename(f'risk')

        return port_risk


