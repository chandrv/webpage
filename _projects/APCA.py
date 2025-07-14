# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:47:55 2024

@author: chandrv
"""

#%%
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import os
os.chdir(r'C:\Users\chandrv\Risk_Dashboards\ge_dashboard')

from RiskEngine.RiskPortfolio import *
from RiskEngine.PortfolioReturns import *

from RiskEngine.Database import Database
import sqlalchemy as sa
import os
import pandas as pd
import numpy as np
from ML.MLEquityValuation import *

from pandas.tseries import offsets

from pandas.tseries.offsets import DateOffset
#import optuna.integration.lightgbm as opt
import datetime as dt
import regex as re
from tqdm import tqdm
from statsmodels.api import OLS
from numpy.linalg import eig, inv, pinv

from scipy.stats import ttest_1samp

from numpy.linalg import eig
#%%

ep = EquityPortfolio(dt.datetime(2024,3,15))
ep.set_benchmark('MXWD')
isin_list = list(ep.benchmark_portfolio.index)
weight_list = list(ep.benchmark_portfolio['weight'])

pr = PortfolioReturns(start_date=dt.datetime(2023,3,16),end_date=dt.datetime(2024,3,16))
pr.set_custom_portfolio(isin_list=isin_list,weight_list=weight_list,missing_thresh=0.85)

asset_factor_ret_raw = pr.get_asset_factor_ts()




asset_factor_ret = asset_factor_ret_raw.dropna(subset=['specific_return'])
#asset_factor_ret = asset_factor_ret[asset_factor_ret['total_return'] != 0]
asset_factor_ret.drop_duplicates(inplace=True)

tot_ret = asset_factor_ret['total_return'].unstack()
#f = ((tot_ret==0).sum()/len(tot_ret)) >= 0.15
residuals = asset_factor_ret['specific_return'].unstack()

#%%
def filter_outliers(R_xts,q=0.95,missing=0.2):
    f1 = R_xts.max() <= R_xts.max().quantile(q)
    f2 = ((R_xts == 0).sum()/len(R_xts)) <= missing
    R_filtered = R_xts.loc[:,f1 & f2]
    return R_filtered


def multivariate_r2(y,yHat):
    r_squared = 1 - (np.sum((y - yHat)**2, axis=0) / np.sum((y - y.mean(axis=0))**2, axis=0))
    return r_squared

def UseAPCA(R_xts,k=5,corr=False,refine=True):
    #https://github.com/AvinashAcharya/factorAnalytics/blob/master/R/fitSfm.R
    R_mat = R_xts.values # TxN
    n = R_mat.shape[1]
    obs = R_mat.shape[0]
    
    # demean TxN matrix of returns
    R_mat_d = R_mat - R_mat.mean(axis=0)
    
    # TxT return covariance matrix
    Omega_T = np.dot(R_mat_d, R_mat_d.T)/n
    
    if corr: 
        #Use corr instead of cov. Variables with the highest variance will dominate PCA; using corr scales them to unit variance first 
        #However, this may be justified if a volatile asset is more interesting for some reason and volatility information shouldn't be discarded. On the other 
        #using the correlation matrix standardizes the variables and makes them comparable, avoiding penalizing variables with less dispersion. 
        Omega_T = np.corrcoef(Omega_T)
    
    # get eigen decomposition
    eig_val, eig_vec = eig(Omega_T)
    
    # get TxK factor realizations
    X = eig_vec[:, :k] # TxK
    f = pd.DataFrame(X, index=R_xts.index, columns=["F."+str(i+1) for i in range(k)])
    
    # invert 1st principal component if most values are negative
    if f.iloc[:,0].median() < 0:
        f.iloc[:,0] *= -1
    
    # LS time series regression to get B: NxK matrix of factor loadings
    
    fC = sm.add_constant(f)
    
    asset_fit = OLS(R_xts, fC).fit()
    yHat = asset_fit.predict(fC)
    yHat.columns = R_xts.columns
    B = asset_fit.params.T
    alpha = asset_fit.params.iloc[0]
    
    
    # estimate residual standard deviations
    resid_sd = asset_fit.resid.std()
    
    if refine:
        #R_mat_rescaled = (R_mat_d.T)/resid_sd
        R_mat_rescaled = R_mat_d.T/resid_sd.values[:,None]
        Omega_T = np.dot(R_mat_rescaled.T, R_mat_rescaled)/n
        if corr:
            Omega_T = np.corrcoef(Omega_T)
        eig_val, eig_vec = eig(Omega_T)
        
        X = eig_vec[:, :k]
        f = pd.DataFrame(X, index=R_xts.index, columns=["F."+str(i+1) for i in range(k)])
        
        if f.iloc[:,0].median() < 0:
            f.iloc[:,0] *= -1
        
        fC = sm.add_constant(f)
        
        asset_fit = OLS(R_xts, fC).fit()
        yHat = asset_fit.predict(fC)
        yHat.columns = R_xts.columns
        B = asset_fit.params.T
        alpha = asset_fit.params.iloc[0]
        resid_sd = asset_fit.resid.std()
    
    # compute factor model return covariance: NxN
    B = B.iloc[:,1:]
    
    B.index = R_xts.columns
    
    Omega_fm = np.dot(np.dot(B, f.cov()), B.T) + np.diag(np.square(resid_sd))
    
    
    # compute factor mimicking portfolio weights
    mimic = np.dot(pinv(R_mat), f)
    mimic = mimic/mimic.sum(axis=0)
    
    # extract r2, residuals
    resid_xts = asset_fit.resid
    r2 = multivariate_r2(R_xts.values,yHat.values)
    r2 = pd.Series(r2,index=R_xts.columns,name='r2')
    
    # return dictionary
    return {"asset_fit": asset_fit, "yHat":yHat,"k": k, "factors": f, "loadings": B, "alpha": alpha, "r2":r2,
            "resid_sd": resid_sd, "residuals": resid_xts, "Omega": Omega_fm, "eigen": eig_val, 
            "mimic": mimic}

from scipy.stats import ttest_1samp

def UseAPCA_ck(R_xts, max_k, refine, sig, corr):
    n = R_xts.shape[1]
    obs = R_xts.shape[0]
    idx = np.arange(1, obs, 2)
    
    # dof-adjusted squared residuals for k=1
    fit = UseAPCA(R_xts, 1, refine, corr)
    eps2 = (fit['residuals']**2) / (1 - 2/obs - 1/n)
    
    for k in range(2, max_k + 1):
        f = fit
        mu = eps2.iloc[idx].mean(axis=1)
        # dof-adjusted squared residuals for k
        fit = UseAPCA(R_xts, k, refine, corr)
        eps2_star = (fit['residuals']**2) / (1 - (k + 1)/obs - k/n)
        mu_star = eps2_star.iloc[idx].mean(axis=1)
        # cross sectional differences in sqd. errors btw odd & even time periods
        delta = mu - mu_star
        # test for a positive mean value for Delta
        if ttest_1samp(delta, 0, alternative='greater').pvalue > sig:
            return f
        eps2 = eps2_star
    return fit


def UseAPCA_bn(R_xts, max_k, refine, corr):
    n = R_xts.shape[1]
    obs = R_xts.shape[0]
    # initialize sigma
    sigma = np.empty(max_k)
    
    for k in range(1, max_k + 1):
        # fit APCA for k factors
        fit = UseAPCA(R_xts, k, refine, corr)
        # get cross-sectional average of residual variances
        sigma[k - 1] = (fit['resid_sd']**2).mean()
    
    idx = np.arange(1, max_k + 1)
    # Preferred criteria PC_p1 and PC_p2
    PC_p1 = sigma[idx - 1] + idx * sigma[-1] * (n + obs) / (n * obs) * np.log((n * obs) / (n + obs))
    PC_p2 = sigma[idx - 1] + idx * sigma[-1] * (n + obs) / (n * obs) * np.log(min(n, obs))
    
    if np.argmin(PC_p1) != np.argmin(PC_p2):
        print("PC_p1 and PC_p2 did not yield the same result. The smaller one was used.")
    k = min(np.argmin(PC_p1), np.argmin(PC_p2)) + 1
    return UseAPCA(R_xts, k, refine, corr)

def coef_sfm(fit):
    coef_mat = pd.concat([fit['alpha'], fit['loadings']], axis=1)
    coef_mat.columns = ['(Intercept)'] + list(coef_mat.columns[1:])
    return coef_mat

def fitted_sfm(fit):
    return fit['data'] - fit['residuals']

def residuals_sfm(fit):
    return fit['residuals']


#%%
from statsmodels.tsa.stattools import coint
def get_spread(x,y):

    xC = sm.add_constant(x)
    regr = OLS(y,xC).fit()    

    beta = regr.params[1]
    alpha = regr.params[0]
    spread = y - x*beta - alpha
    return spread


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
#%%

X = tot_ret.copy()
X = residuals.copy()
X = X.fillna(0)
R_xts = filter_outliers(X,q=0.95,missing=0.15)

test = UseAPCA_ck(R_xts,5,sig=0.05,refine=True,corr=True)


tmp = test['yHat'].stack()
tmp.index.names = ['date','security__id_isin']
tmp = tmp.rename('common_residual')
adj_factor_ret = asset_factor_ret.join(tmp)
adj_factor_ret['true_resid'] = adj_factor_ret['specific_return']-adj_factor_ret['common_residual']

#%%

data = test['yHat'].copy()
fitted_px = (1+data).cumprod()
fitted_px = fitted_px.loc[:,fitted_px.columns.isin(ge.portfolio.index)]
scores, pvalues, pairs = find_cointegrated_pairs(fitted_px)

pv = pd.DataFrame(pvalues,index=fitted_px.columns,columns=fitted_px.columns).stack()
pv = pv[pv!=1]
pv = pv[pv>=0.99]

#%%
ge = EquityPortfolio(dt.datetime(2024,3,15))
ge.set_ocm_portfolio('GE Core')


#%%


ticks = ep.isin_to_ticker(R_xts.columns,truncate=True)
B = test['loadings']

from scipy.spatial.distance import squareform

import scipy.cluster.hierarchy as hr
import riskfolio as rp
import riskfolio.src.AuxFunctions as af
import riskfolio.src.DBHT as db
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
c = test['loadings'].copy()
c.drop('ticker',axis=1,inplace=True)
c = cosine_distances(c)
c = pd.DataFrame(c,index=B.index,columns=B.index)

#dist = dist.to_numpy()
#dist = pd.DataFrame(dist, columns=x.columns, index=x.index)
p_dist = squareform(c, checks=True)
clustering = hr.linkage(p_dist, method='ward', optimal_ordering=True)

clustering_inds = hr.fcluster(clustering, 10, criterion="maxclust")
tmp = ep.benchmark_portfolio.copy()
tmp.reindex(B.index)
tmp=tmp.reindex(B.index)
tmp['ticker'] = B['ticker']
tmp['cluster'] = clustering_inds

#%%


gd = gsd.GSData(analysis_date=dt.datetime(2024,3,15))
gd.get_basket_coverage()


gd.filter_key_baskets(asia_baskets=False)
gd.baskets = gd.baskets[gd.baskets['name'] != 'GS Southbound Favorites']
gd.get_thematic_ts(start_date=dt.datetime(2023,3,16),end_date=dt.datetime(2024,3,16))

#%%
basket_rets = gd.index_ts.pct_change()
f_tmp = test['factors'].copy()
basket_rets = basket_rets.reindex(f_tmp.index)

fC = sm.add_constant(f_tmp)

basket_fit = OLS(basket_rets, fC).fit()
basket_ldg = basket_fit.params.T
basket_ldg.index = basket_rets.columns
basket_fitted = basket_fit.predict(fC)
basket_fitted.columns = basket_rets.columns
#%%


fitted = test['yHat'].copy() #common component in residuals
fitted = fitted.loc[:,fitted.columns.isin(ge.portfolio.index)]
c = fitted.corr()

pc = PortfolioCluster(fitted)

date_idx = pc.resample_dates(fitted,start_date=dt.datetime(2023,3,17),
                             time_step=3,
                             min_periods=3*21)

corr_resample = pc.get_ewma_corr_series(date_idx,half_life_months=3,denoise=False)



#%%corr_mat = pc.get_ewma_corr(half_life_months=6,denoise=False)
#%%clusters,ax = pc.cluster_hierarchy(corr_mat,linkage='ward',show_clusters=True)

#%%

B_ge = test['loadings'][test['loadings'].index.isin(ge.portfolio.index)]


ok = fitted.groupby(fitted.columns.map(clusters),axis=1).mean()

c_cum = (1+ok).cumprod()

clusters = c_cum.columns

gd.index_ts = (1+basket_fitted).cumprod()

theme_dict = {x:gd.hedge_portfolio(c_cum[x]) for x in clusters}

#%%

def AR(a):
    return AbsorptionRatio(a).estimate()

def rolling_pipe(dataframe, window, fctn):
    return pd.Series([dataframe.iloc[i-window: i].pipe(fctn) 
                      if i >= window else None 
                      for i in range(1, len(dataframe)+1)],
                     
                     
                     index = dataframe.index) 
from frds.measures import AbsorptionRatio



ok = R_xts.pipe(rolling_pipe, 60, lambda x: AR(x))