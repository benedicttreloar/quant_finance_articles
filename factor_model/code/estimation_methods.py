import numpy as np
import pandas as pd
import statsmodels.api as sm

def ols(reg_returns, reg_exposures, reg_weights):

    '''Fit OLS model using statsmodels'''

    # Fit model using statsmodels
    mod = sm.OLS(reg_returns, reg_exposures)
    res = mod.fit()

    # Calculate factor returns
    factor_returns = res.params.transpose()
    idio_returns = res.resid

    # Collect model power data
    rsq = res.rsquared
    tstat = res.tvalues

    return factor_returns, idio_returns, rsq, tstat

def wls(reg_returns, reg_exposures, reg_weights):

    '''Fit OLS model using statsmodels'''

    # Fit model using statsmodels
    mod = sm.WLS(reg_returns, reg_exposures, reg_weights)
    res = mod.fit()

    # Calculate factor returns
    factor_returns = res.params.transpose()
    idio_returns = res.resid

    # Collect model power data
    rsq = res.rsquared
    tstat = res.tvalues

    return factor_returns, idio_returns, rsq, tstat