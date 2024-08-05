import numpy as np
import pandas as pd
import estimation_methods

class Model:

    def __init__(self, market, model_name:str):

        """ Class: estimate model structure and return model data

        Parameters:
        ------------
        market : SimulatedMarket (an input market structure with stock returns and exposures
        model_name : str (a model name to refer to the model by)
        """

        self.__market = market
        self.__model_name = model_name

    @property
    def market(self):
        return self.__market

    @property
    def model_name(self):
        return self.__model_name

    @property
    def fit(self,
            excluded_factors: list = None,
            method='equal_weight'
            ):

        # Fit factor model
        if method in ('equal_weight'):
            self._iterate_model_fitting(method=method,
                                        excluded_factors=excluded_factors)
        else:
            raise NotImplementedError

    def _iterate_model_fitting(self, method, excluded_factors: list = None):

        #### Fit a fundamental factor model of market returns

        # Store excluded factors
        self.__excluded_factors = excluded_factors

        # Estimate model
        model_factors_dict = {}
        model_idio_dict = {}
        model_rsq_dict = {}
        model_tstat_dict = {}

        input_returns = self.market.in_sample_returns.copy()

        # Period by period, estimate within-period factor returns and idiosyncratic returns to go with it
        for date in input_returns.columns:

            # Generate input data for model estimation
            reg_returns = input_returns[date]
            reg_exposures = self._get_exposures(date)

            # Estimate factor and idiosyncratic returns
            if method == 'equal_weight':
                reg_weights = None
                mod_factor_returns, mod_idio_returns, mod_rsq, mod_tstat = estimation_methods.ols(
                    reg_returns, reg_exposures, reg_weights)
            elif method == 'sqrt_mktcap':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Collect factor returns and model power data
            model_factors_dict[date] = mod_factor_returns
            model_idio_dict[date] = mod_idio_returns

            # Collect model power data
            model_rsq_dict[date] = mod_rsq
            model_tstat_dict[date] = mod_tstat

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

    def _get_exposures(self, date, excluded_factors: list = None):

        # Generate explicit excluded factor list
        if excluded_factors is not None:
            excluded_factors_explicit = self._get_excluded_factor_list(excluded_factors)
        else:
            excluded_factors_explicit = []

        # Generate exposures
        output_exposures = self.market.exposures.drop(excluded_factors_explicit, axis=1).copy()

        return output_exposures


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
