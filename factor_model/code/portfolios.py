# Generate a set of random portfolios

import numpy as np
import pandas as pd
from tqdm import tqdm

# Portfolio Structure Defaults
DEFAULT_PORTFOLIO_SIZE = 60
DEFAULT_PORTFOLIO_NUMBER = 200

class Portfolio:

    def __init__(
            self,
            market,
            port_id
    ):

        """ Generate a randomly generated portfolios for a given market

        Parameters:
        ---------
        :param market: either a SimulatedMarket object or an int
        :param port_id: the id of the given portfolio

        Returns
        ---------
        self: returns an instance of the Portfolios class
        """

        self.__market = market
        self.__port_id = port_id

    def reset(self):

        ''' Resets the portfolio weights '''
        self.__weights = pd.DataFrame(0., index=self.security_ids, columns=self.port_ids)

    def generate_portfolio(
            self,
            weight_method='equal_weight',
            port_size=DEFAULT_PORTFOLIO_SIZE
    ):

        ''' Creates a set of portfolios with the desired set of random weights structure

        Parameters:
        ---------
        :param weight_method: the methodology for generating weights
        :param port_size: the number of securities to hold in each portfolio
        :param port_ids: The vector of IDs to use for each portfolio

        Returns
        ---------
        self: returns an instance of the Portfolios class
        '''

        self.reset()
        if weight_method == 'equal_weight':
            for port in tqdm(self.port_ids):
                self.__weights.loc[
                    np.random.choice(self.security_ids, self.port_size, replace=False).tolist()] = 1 / self.port_size
        else:
            raise NotImplementedError(f'{weight_method} not implemented')

        return self.weights

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
    def port_id(self):
        return self.__port_id

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
