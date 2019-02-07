#Creating testing and training data sets for AR realizations

import numpy as np
import statsmodels
from statsmodels.tsa import arima_process


def AR_data_creation(p, n_samples, r_len, verbose = False):
    """
    Creates a data set of AR realizations of order 0 - p each containing n_samples
    
    Arguements:
        p - The maximum AR order to include in the data set
        n_samples - The number of samples n per order p to create
        r_len - The length of the realization
        verbose - when set to True prints done after an order of p is complete

    Returns:
    A numpy array of shape (p,n_samples,1,r_len) which is the full AR training set data.
    """

    #Intializes data set shape
    AR_data = np.zeros((p, n_samples, 1,r_len))
    data_type = "base_AR"
    data_creation_loop(AR_data, p, n_samples, r_len, verbose, data_type)
    return AR_data


def CNN_AR_data_creation(p, n_samples, r_len, verbose = False):
    """
    Creates a data set of AR realizations of order 0 - p each containing n_samples
    
    Arguements:
        p - The maximum AR order to include in the data set
        n_samples - The number of samples n per order p to create
        r_len - The length of the realization
        verbose - when set to True prints done after an order of p is complete

    Returns:
    A numpy array of shape (p,n_samples,1,r_len) which is the full AR training set data.
    """
    if r_len % 2 == 1:
        r_len -= 1
    AR_data = np.zeros((p, n_samples, int(r_len/2),int(r_len/2)))
    r_len = int(r_len/2)
    data_type = "square_AR"
    data_creation_loop(AR_data, p, n_samples, r_len, verbose, data_type)
    return AR_data


def acfpacf_creation(p, n_samples, r_len, lags = 25,verbose = False):
    """
    Creates a data set of acf and pacf of legnth equal to number of lags based of 
    a r_len realizations of order 0 - p each containing n_samples
    
    Arguements:
        p - The maximum AR order to include in the data set
        n_samples - The number of samples n per order p to create
        r_len - The length of the realization
        verbose - when set to True prints done after an order of p is complete

    Returns:
    A numpy array of shape (p,n_samples,2, lags) which is the full AR training set data.
    """

    AR_data = np.zeros((p, n_samples, 2, lags))
    data_type = "acfpacf"
    data_creation_loop(AR_data, p, n_samples, r_len, verbose, data_type, lags)
    return AR_data


def data_creation_loop(AR_data, p, n_samples, r_len, verbose, data_type, lags = 25):
    """
    Mutates AR_data to create AR data set of desired type

    Arguements:
        AR_data - numpy array that will be mutated to hold the data
        p - The maximum AR order to include in the data set
        n_samples - The number of samples n per order p to create
        r_len - The length of the realization
        verbose - when set to True prints done after an order of p is complete
        data_type - the type of the data that is desired
        lags - number of lags wanted, only needed for acfpacf data    
    """

    #iterates for each order less than p
    for p_index in range(p):

        #use while loop to keep iterating so that we have n_samples of AR realizations that causal 
        i = 0 
        while n_samples - 1 > i:
            coefs = (2 * np.random.random(p_index)) - 1
            if p_index > 1:
                inout = np.random.randint(0, high = 2, size= p_index - 1)
                inout = np.r_[tuple((inout,1))]
            else:
                inout = 1
            coefs = coefs * inout        
            coefs = coefs / (sum(abs(coefs)) + 0.0001)
            coefs = np.r_[tuple((1, -coefs))]
            if coefs[-1] < .15:
                pass 
            else:
                i += 1
                if data_type == "base_AR":
                    AR_data[p_index][i][0] = statsmodels.tsa.arima_process.arma_generate_sample(ar = coefs, ma = [1], nsample = r_len, sigma = 1, burnin = 24)

                if data_type == "square_AR":
                    realization = statsmodels.tsa.arima_process.arma_generate_sample(ar = coefs, ma = [1], nsample = r_len, sigma = 1, burnin = 24)
                    for j in range(int(r_len/2)):
                        AR_data[p_index][i][j] = realization[j:int(r_len/2) + j]

                if data_type == "acfpacf":
                    realization = statsmodels.tsa.arima_process.arma_generate_sample(ar = coefs, ma = [1], nsample = r_len, sigma = 1, burnin = 24)
                    AR_data[p_index][i][0] = statsmodels.tsa.stattools.acf(realization, nlags = lags - 1)
                    AR_data[p_index][i][1] = statsmodels.tsa.stattools.pacf(realization, nlags = lags - 1)

        if verbose:
            print("Done:", p_index)
    return AR_data


