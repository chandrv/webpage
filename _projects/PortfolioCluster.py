# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:08:43 2023

@author: chandrv
"""
#from orm_toolbox import PortfolioHub
import os
import pandas as pd
import numpy as np
from pandas.tseries import offsets
import scipy.stats as sps
from statsmodels.stats.moment_helpers import corr2cov,cov2corr
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import plotly.express as px

import plotly.graph_objects as go
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt

import scipy.cluster.hierarchy as hr
import riskfolio as rp
import riskfolio.src.AuxFunctions as af
import riskfolio.src.DBHT as db
from scipy.spatial.distance import squareform

from RiskEngine.Factors import *
from RiskEngine.RiskPortfolio import *
#%%

            
def flatten_list(x):
    return [item for sublist in x for item in sublist]

class PortfolioCluster():
    def __init__(self,asset_returns=None):
        self.asset_returns = asset_returns
        
    def filter_missing_returns(self,missing_thresh = 0.2):
        asset_returns = self.asset_returns
        notnan = asset_returns.notnull().sum()
        
        if missing_thresh < 1: #interpret as percent
            notnan = notnan/len(asset_returns)
    
        notnan = notnan >= missing_thresh #check if sufficient 
        asset_filtered = asset_returns.loc[:,asset_returns.columns[notnan]]
        
        self.asset_returns = asset_filtered
        
    def get_ewma_weights(self,n,halflife):
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        weights = np.power(1 - alpha, np.arange(n)[::-1])
        weights /= weights.sum()
        return weights
        
    def fast_ewma_pairwise(self,asset_returns, halflife,return_corr=False):
        n=len(asset_returns)
        weights = self.get_ewma_weights(n=n,halflife=halflife)
 
        # Convert DataFrame to a NumPy array with a masked array to handle NaNs
        masked_array = np.ma.masked_invalid(asset_returns.values)
    
        # Calculate the weighted mean for each asset
        asset_means = np.ma.average(masked_array, axis=0, weights=weights)
    
        # Compute the pairwise exponentially weighted moving average correlation
        diff_from_mean = masked_array - asset_means
        weighted_diff = diff_from_mean * np.sqrt(weights[:, None])
        cov_matrix = np.ma.dot(weighted_diff.T, weighted_diff) / n
        
        if return_corr:
        
            # Calculate the standard deviation for each asset
            asset_std = np.ma.sqrt(np.diag(cov_matrix))
            
            # Compute the correlation matrix from the covariance matrix
            correlation_matrix = cov_matrix / np.outer(asset_std, asset_std)
            
            # Convert the correlation matrix back to a DataFrame
            return pd.DataFrame(correlation_matrix.data, index=asset_returns.columns, columns=asset_returns.columns)
        
        else:
            
            return  pd.DataFrame(cov_matrix.data, index=asset_returns.columns, columns=asset_returns.columns)


    def fast_ewma_std(self,asset_returns, halflife):
        n=len(asset_returns)
        weights = self.get_ewma_weights(n=n,halflife=halflife)
 
    
        # Convert DataFrame to a NumPy array with a masked array to handle NaNs
        masked_array = np.ma.masked_invalid(asset_returns.values)
    
        # Calculate the weighted mean for each asset
        asset_means = np.ma.average(masked_array, axis=0, weights=weights)
    
        # Compute the exponentially weighted moving average variance for each asset
        diff_from_mean = masked_array - asset_means
        squared_diff = np.square(diff_from_mean)
        ewma_var = np.ma.average(squared_diff, axis=0, weights=weights)
    
        # Calculate the exponentially weighted moving average standard deviation for each asset
        ewma_std = np.sqrt(ewma_var)
    
        # Convert the standard deviation to a DataFrame
        return pd.Series(ewma_std.data, index=asset_returns.columns)

    def get_ewma_cov(self,half_life_months=6):
        cov = self.fast_ewma_pairwise(self.asset_returns,halflife=half_life_months*21)
        cov.index.names = ['name_0']
        cov.columns.name = 'name'
        return cov
    
    def get_ewma_corr(self,half_life_months=6,denoise=False,**kwargs):
        corr = self.fast_ewma_pairwise(self.asset_returns,halflife=half_life_months*21,return_corr=True)
        if denoise:
            corr = self.denoise_matrix(corr,kwargs.get('kind','fixed'),kwargs.get('bWidth',0.01),kwargs.get('detone',True)) 
        corr.index.names = ['name_0']
        corr.columns.name = 'name'
        return corr
    
    def get_ewma_corr_series(self,date_idx,half_life_months=6,denoise=False,**kwargs):
        corr_series = []
        for i in date_idx:
            corr_i = self.fast_ewma_pairwise(self.asset_returns.loc[:i],halflife=half_life_months*21,return_corr=True)
            
            if denoise:
                corr_i = self.denoise_matrix(corr_i,kwargs.get('kind','fixed'),kwargs.get('bWidth',0.01),kwargs.get('detone',True)) 
            
            corr_i['date'] = i
            corr_series.append(corr_i)
            
        corr_series = pd.concat(corr_series)
        corr_series = corr_series.set_index('date',append=True).reorder_levels(['date',corr_series.index.name])
        return corr_series
    
    def get_rolling_ewma_cov(self,half_life_months=6):
        rolling_cov =  self.asset_returns.ewm(halflife=half_life_months*21,min_periods=half_life_months*21).cov()#.dropna()
        rolling_cov.index.names = ['date','name_0']
        rolling_cov.columns.name = 'name'
        rolling_cov = rolling_cov.groupby(level=0).apply(lambda x: pd.DataFrame(x.values,index=list(rolling_cov.columns),columns=list(rolling_cov.columns)))
        self.rolling_cov = rolling_cov

    
    def get_rolling_avg_corr(self,half_life_months=6):
        self.get_rolling_ewma_cov(half_life_months)
       #rolling_cov = rolling_cov.drop(rolling_cov.index.get_level_values(0)[0])
        rolling_cov = self.rolling_cov
        rolling_corr = rolling_cov.groupby(level=0).apply(lambda x: pd.DataFrame(cov2corr(x),index=list(rolling_cov.columns),columns=list(rolling_cov.columns)))
        rolling_corr = rolling_corr.dropna()
        rolling_corr.index.names = ['date','name_0']
        rolling_corr.columns.name = 'name'
        self.rolling_corr = rolling_corr
        
    
    
    def get_rolling_portfolio_avg_corr(self,weights,halflife_months=6):
        
        if not hasattr(self,'rolling_cov'):
            self.get_rolling_ewma_cov(halflife_months)
        
        avg_corr_ts = self.rolling_cov
        
        #check = avg_corr_ts.groupby(level=1).last().isna().any()
        #drop_idx = check[check]
        #if len(drop_idx) > 0:
        #    drop_idx = drop_idx.index
        #    avg_corr_ts = avg_corr_ts.drop(drop_idx,axis=1)
        #    avg_corr_ts = avg_corr_ts.drop(drop_idx,axis=0)
        
        avg_corr_ts = avg_corr_ts.dropna().groupby(level=0).apply(lambda x: self.portfolio_avg_corr(x,weights))
        
        return avg_corr_ts
    
    def portfolio_avg_corr(self,cov,wts):
        
        try:
            if len(cov.index.names) > 1:
                cov = cov.droplevel(0)
                    
                wts = wts.reindex(cov.index)
                wts = wts/wts.sum()
                wts = wts.fillna(0)
                
                    
            n = np.sqrt(wts@cov@wts.T)
            d = (wts@np.diag(cov))**0.5
            return (n/d)**2
        except:
            return np.nan
    
    def construct_edgelist(self,corr_mat,thresh=None):
        links = corr_mat.stack().reset_index()
        links.rename(columns={0:'corr'},inplace=True)
        edgelist = links.loc[(links['name_0'] != links['name'])]
        
        if thresh:
            edgelist=edgelist.loc[ (edgelist['corr'] >= thresh)]
            
        return edgelist
    
        
    def construct_pmfg(self,corr_mat):
        edgelist = self.construct_edgelist(corr_mat,thresh=None)
        edgelist = edgelist.sort_values(by=['corr'],ascending=False) #presort by distance
        G=nx.from_pandas_edgelist(edgelist, 'name_0', 'name',edge_attr='corr')
        
        pmfg = nx.Graph()
        nb_nodes = len(G.nodes)
        for edge in tqdm(list(G.edges(data=True))):
            pmfg.add_edge(edge[0], edge[1],corr=edge[2]['corr'])
            if not nx.is_planar(pmfg):
                pmfg.remove_edge(edge[0], edge[1])
                
            if len(pmfg.edges()) == 3*(nb_nodes-2):
                break
        return pmfg
            
    def construct_mst(self,corr_mat):
        edgelist = self.construct_edgelist(corr_mat,thresh=None)
        edgelist['dist'] = 1-edgelist['corr']
        G=nx.from_pandas_edgelist(edgelist, 'name_0', 'name',edge_attr=['dist','corr'])
        #filtered_nodes = [x for x in corr_mat.index if x not in list(G.nodes)]
        #G.add_nodes_from(filtered_nodes)"

        T=nx.minimum_spanning_tree(G,weight='dist')
        return T
    
    def construct_graph(self,corr_mat,thresh):
        edgelist = self.construct_edgelist(corr_mat,thresh)
        G=nx.from_pandas_edgelist(edgelist, 'name_0', 'name',edge_attr='corr')
        filtered_nodes = [x for x in corr_mat.index if x not in list(G.nodes)]
        G.add_nodes_from(filtered_nodes)
        return G
    
    def denoise_matrix(self,corr_mat,kind='fixed',bWidth=0.01,detone=False):
        
        if type(corr_mat.index) == pd.MultiIndex:
           corr_mat = corr_mat.droplevel(0,axis=0) 
        
        T, N = self.asset_returns.shape
        q = T / N
        eVal, eVec = af.getPCA(corr_mat)
        eMax, var = af.findMaxEval(np.diag(eVal), q, bWidth)
        nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)
        corr_d = af.denoisedCorr(eVal, eVec, nFacts, kind=kind)
        corr_d = pd.DataFrame(corr_d, index=corr_mat.index, columns=corr_mat.columns)
        
        if detone:
            mkt_comp = 1 #Remove the first principal component
            eVal_ = eVal[:mkt_comp, :mkt_comp]
            eVec_ = eVec[:, :mkt_comp]
            corr_ = np.dot(eVec_, eVal_).dot(eVec_.T)
            corr_d = corr_d - corr_
            #rescale 
            c_diag = np.sqrt(np.diag(corr_d))
            corr_d = corr_d/np.outer(c_diag,c_diag)

        
        return corr_d
    
    def denoise_rolling_corr(self,kind='fixed',detone=False):
        denoised_rolling = self.rolling_corr.groupby(level=0).apply(lambda x: self.denoise_matrix(x,kind=kind,detone=detone))
        self.rolling_corr = denoised_rolling
        
    def set_graph(self,corr_mat,graph_type='filtered',thresh=None):
        
        corr_mat.index.names = ['name_0']
        corr_mat.columns.name = 'name'
        
        if graph_type == 'filtered':
            G = self.construct_graph(corr_mat,thresh)
        elif graph_type == 'mst':
            G = self.construct_mst(corr_mat)
        elif graph_type == 'pmfg':
            G = self.construct_pmfg(corr_mat)
            
        self.graph = G
    
    def partition_graph(self,method='louvain'):
      
        G = self.graph
        
        if method == 'louvain':\
            part = nx_comm.louvain_communities(G, weight='corr',seed=0)
            #community_louvain.best_partition(G, weight='corr')
            
        elif method == 'propagation':
          #  part = list(nx_comm.label_propagation.label_propagation_communities(G))
            part = list(nx_comm.label_propagation.asyn_lpa_communities(G,weight='corr'))
            
        elif method == 'modularity':
            part = list(nx_comm.modularity_max.greedy_modularity_communities(G,weight='corr'))

            
        #Inverse map so that we have a dictionary of ID:group membership
        groups = []
        
        for idx,group in enumerate(part):
            if len(group) > 1:
                groups.extend([(v,idx) for v in group])
            else:
                groups.extend([(list(group)[0],-1)])
                
        return dict(groups)
    
    def partc(self,c,community_method='louvain',thresh=0.5):
        c = c.droplevel(0)
        self.set_graph(c,'filtered',thresh)
        if sum(dict(self.graph.degree(weight='corr')).values()) > 0: #If there's no links above that thresh
            inv_map = self.partition_graph(method=community_method)#,filtered_nodes=None)
        else: 
           inv_map = dict([(k,-1) for k in list(self.graph.nodes)])
        return pd.Series(inv_map)
    
    def hc(self,c,maxclust=10,return_partition=False,k=None,method='ward'):
        #function to run a fast hclust on 
        if type(c.index) == pd.MultiIndex:
            c = c.droplevel(0)
            
        dist = np.sqrt(np.clip((1 - c) / 2, a_min=0.0, a_max=1.0))
        np.fill_diagonal(dist.values,0) 
        #dist = dist.to_numpy()
        #dist = pd.DataFrame(dist, columns=x.columns, index=x.index)
        p_dist = squareform(dist, checks=False)
        clustering = hr.linkage(p_dist, method=method, optimal_ordering=True)
        
        if k is None:
            k = af.two_diff_gap_stat(c, dist, clustering, maxclust)
            
        clustering_inds = hr.fcluster(clustering, k, criterion="maxclust")
        
        #check for unclustered
        #a,b = np.unique(clustering_inds,return_counts=True)
        
        #unclustered = a[b==1]
        
        #clustering_inds[np.isin(clustering_inds, unclustered)] = 0
        
        
        idx = list(c.index)
        
        part = {i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)}
        for i, v in enumerate(clustering_inds):
            part[v].append(idx[i])
            
        if return_partition:
            return part
        
        else: #convert to series
                
        
        #clusters = dict(zip(list(corr_mat.index),clustering_inds))
        
            groups = []
            part = list(part.values())
            i = 0
            for idx,group in enumerate(part):
                if len(group) > 1:
                    groups.extend([(v,i) for v in group])
                    i = i+1
                else:
                    groups.extend([(list(group)[0],-1)])
                    
            clusters = dict(groups)
            
            
            return pd.Series(clusters)
        
        
        
    def check_avg_corr(self,corr_df,subset):
        corr_trunc = corr_df.loc[subset,subset]
        avg_corr = corr_trunc.values[np.triu_indices_from(corr_trunc.values,1)].mean()
        return avg_corr
    
    def check_cluster_avg_corrs(self,clst,corr_df):
        clst_corrs = {}
        for k,v in clst.items():
            if len(v) > 1:
                avg_corr = self.check_avg_corr(corr_df,v)
            else:
                avg_corr = 1
               
            clst_corrs[k] = avg_corr
        return clst_corrs
            
    def filter_clusters(self,clusters,c,cutoff=0.5):
        
        cluster_corrs = self.check_cluster_avg_corrs(clusters,c)
        keep_clusters = [k for k,v in cluster_corrs.items() if v>=cutoff]
        
        clustered_assets = [v for k,v in clusters.items() if k  in keep_clusters]
        remaining_assets = flatten_list([v for k,v in clusters.items() if k not in keep_clusters])
        
        return clustered_assets,remaining_assets
        
    def hc_recursive(self,corr_df,cutoff=0.5):
        
        max_clusters = len(corr_df)+1
        
        _corr = corr_df.copy()
        
        recursive_clusters = []
                
        remaining_assets = list(_corr.index)
        
    #  n = filter_loop(_corr)
        for i in range(2,max_clusters):
            part = self.hc(_corr,return_partition=True,k=i)
            
            clustered_assets,remaining_assets = self.filter_clusters(part,_corr,cutoff)
            
            if len(clustered_assets) > 0:
                recursive_clusters.extend(clustered_assets)
                _corr = _corr.loc[remaining_assets,remaining_assets]
            if len(remaining_assets) == 0:
                break
        
        singletons = [x for x in corr_df.index if x not in flatten_list(recursive_clusters)]
        
        if len(singletons) > 1:
            for s in singletons:
                recursive_clusters.append([s])
                          
        recursive_clusters = dict(zip(range(len(recursive_clusters)),recursive_clusters))
        
        return recursive_clusters
        
    def subset_matrix_dates(self,rolling_corr_mat,date_idx):
        
        rolling_corr_mat = rolling_corr_mat[rolling_corr_mat.index.get_level_values(0).isin(date_idx)]
        return rolling_corr_mat

    def resample_dates(self,df,start_date=None,end_date=None,time_step=6,min_periods=None):
        if min_periods:
            df = df.iloc[min_periods:]
        valid_dates = df.index.get_level_values(0)
        
        if start_date is None:
            start_date = valid_dates[0]
        if end_date is None:      
            end_date = valid_dates[-1]
        

        freq = '{}BMS'.format(time_step)
        
        date_idx = pd.date_range(start_date,end_date,freq=freq,inclusive='both')
        date_idx = list(date_idx)
        
        if (end_date-date_idx[-1])/np.timedelta64(1,'M') > time_step/2: #if more than half the time step, extend
            date_idx.extend([end_date])
        else: #if too close, replace
            date_idx[-1] = end_date
        date_idx = [pd.to_datetime(x) for x in date_idx]
        
        #validate dates
        date_idx = [min(valid_dates, key=lambda d: abs(d - date)) for date in date_idx]
        
        return date_idx
    

    def cluster_ts(self,rolling_corr_mat,method='hierarchy',**kwargs):
       # if not hasattr(self,'rolling_cov'):
       #     self.get_rolling_avg_corr(halflife_months)

        if method == 'hierarchy':
            cluster_series = rolling_corr_mat.groupby(level=0).apply(lambda x: self.hc(x))
            #cluster_series = cluster_series-1
        elif method == 'network':
            cluster_series = rolling_corr_mat.groupby(level=0).apply(lambda x: self.partc(x,community_method=kwargs.get('community_method','louvain'),thresh=kwargs.get('thresh',0.5)))
        
        cluster_series = cluster_series.unstack()
            
        return cluster_series

    def map_tickers(self,ids):
        label_dict = EquityPortfolio().isin_to_ticker(ids,truncate=True)
        udf = [x for x in ids if x not in label_dict.keys()]
        
        for x in udf:
            label_dict[x] = x
        return label_dict
        
    def cluster_hierarchy(self,corr_mat,linkage='ward',leaf_order=True,max_k=10,k=None,label_dict=None,show_clusters=False,ax=None,width=10,height=5):
        
        if label_dict is None:
            label_dict = self.map_tickers(corr_mat.columns)

        dist = np.sqrt(np.clip((1 - corr_mat) / 2, a_min=0.0, a_max=1.0))
        dist = dist.to_numpy()
        dist = pd.DataFrame(dist, columns=corr_mat.columns, index=corr_mat.index)
        
        if linkage == "DBHT":
            # different choices for D, S give different outputs!
            D = dist.to_numpy()  # dissimilarity matrix
            np.fill_diagonal(D,0)
            S = (1 - dist**2).to_numpy()
            (_, _, _, _, _, clustering) = db.DBHTs(
            D, S, leaf_order=leaf_order
                )  # DBHT clustering

        else:
            p_dist = squareform(dist, checks=False)
            clustering = hr.linkage(p_dist, method=linkage, optimal_ordering=leaf_order)
        
        labels = np.array(corr_mat.index.map(label_dict))
        

        # Ordering clusterings
        permutation = hr.leaves_list(clustering)
        permutation = permutation.tolist()
        
        if k is None:
            k = af.two_diff_gap_stat(corr_mat, dist, clustering, max_k)
        
        clustering_inds = hr.fcluster(clustering, k, criterion="maxclust")
        
        idx = list(corr_mat.index)
        
        part = {i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)}
        for i, v in enumerate(clustering_inds):
            part[v].append(idx[i])
                
        
        #clusters = dict(zip(list(corr_mat.index),clustering_inds))
        
        groups = []
        part = list(part.values())
        i = 0
        for idx,group in enumerate(part):
            if len(group) > 1:
                groups.extend([(v,i) for v in group])
                i = i+1
            else:
                groups.extend([(list(group)[0],-1)])
                
        clusters = dict(groups)
        
        

        if show_clusters:
            
            if ax is None:
                fig = plt.gcf()
                ax = fig.gca()
                fig.set_figwidth(width)
                fig.set_figheight(height)
            else:
                fig = ax.get_figure()
            
            root, nodes = hr.to_tree(clustering, rd=True)
            nodes = [i.dist for i in nodes]
            nodes.sort()
            nodes = nodes[::-1][: k - 1]
            color_threshold = np.min(nodes)
            colors = af.color_list(k)  # color list
            hr.set_link_color_palette(colors)
            
            
            if len(corr_mat) > 250:
                truncate_mode = 'level'
                no_labels = True
            else:
                truncate_mode=None
                no_labels = False
            

            d = hr.dendrogram(
                clustering, color_threshold=color_threshold, above_threshold_color="grey", ax=ax,truncate_mode=truncate_mode,p=len(nodes)-1,no_labels=no_labels)
            
            
           # dict(zip(labels[permutation],d['leaves_color_list']))
            
            cluster_list = list(set(clusters.values()))
            cluster_num = [x for x in cluster_list if x >= 0]
            color_palette = colors[:len(cluster_num)]
            color_palette = dict(zip(cluster_num,color_palette))
            if -1 in cluster_list:
                color_palette[-1] = '#808080'
            #color_palette = list(set(d['color_list']))

            self.color_palette = color_palette            
            
            hr.set_link_color_palette(None)
            
            if no_labels is False:
                ax.set_xticklabels(labels[permutation],rotation=90, ha="center")
            
            i = 0
            for coll in ax.collections[:-1]:  # the last collection is the ungrouped level
                xmin, xmax = np.inf, -np.inf
                ymax = -np.inf
                for p in coll.get_paths():
                    (x0, _), (x1, y1) = p.get_extents().get_points()
                    xmin = min(xmin, x0)
                    xmax = max(xmax, x1)
                    ymax = max(ymax, y1)
                rec = plt.Rectangle(
                    (xmin - 4, 0),
                    xmax - xmin + 8,
                    ymax * 1.05,
                    facecolor=colors[i],  # coll.get_color()[0],
                    alpha=0.2,
                    edgecolor="none",
                )
                ax.add_patch(rec)
                i += 1
        
            #ax.set_yticks([])
            #ax.set_yticklabels([])
            ax.set_ylabel('Cluster Distance',fontsize=8)
            ax.tick_params(axis='y', labelsize=6,length=0)
            ax.tick_params(axis='x', labelsize=6,length=0)
            ax.spines['left'].set_color('grey')
            #for i in {"right", "left", "top", "bottom"}:
            for i in {"right", "top"}:
                side = ax.spines[i]
                side.set_visible(False)

            
            try:
                fig.tight_layout()
            except:
                pass
            
            return clusters, ax
        else:
            return clusters
                        
 
        

    def cluster_graph(self,corr_mat,graph_type='filtered',community_method='louvain',thresh=None,label_dict=None,group_dict=None,height='700px',width='500px'):
        
        self.set_graph(corr_mat,graph_type,thresh)
                #G,pos = construct_graph(corr_mat)
        
        if sum(dict(self.graph.degree(weight='corr')).values()) > 0: #If there's no links above that thresh
            inv_map = self.partition_graph(method=community_method)#,filtered_nodes=None)
        else: 
           inv_map = dict([(k,-1) for k in list(self.graph.nodes)])
           
        if label_dict is None:
            label_dict = EquityPortfolio().isin_to_ticker(corr_mat.columns,truncate=True)
        
        udf = [x for x in corr_mat.columns if x not in label_dict.keys()]
        
        for x in udf:
            label_dict[x] = x
            
        cluster_net = self.visualize_network(inv_map,label_dict,height,width)    
        
        if group_dict:
            cluster_net,legend = self.color_network(cluster_net,group_dict)
        else:
            cluster_net,legend = self.color_network(cluster_net,inv_map)
        
        return inv_map,cluster_net,legend
    
    def visualize_network(self,inv_map,label_dict,height='700px',width='500px'):

        cluster_net = Network(height=height,width=width)
        cluster_net.from_nx(self.graph)
        #cluster_net.show_buttons(filter_=True)
        
        #cluster_net.set_options('{"layout":{"randomSeed":0}}')
        const_options = {"layout":{
            "randomSeed":0
            },
          "nodes": {
            "font": {
              "size": 42
            },
          },
          "configure": {
              "enabled": True, 
              "filter": True},
          "edges": {
            "color": {
              "inherit": True
            },
            "selfReference": {
              "angle": 0.7853981633974483
            },
            "smooth": {
              "forceDirection": None
            }
          },
          "interaction": {
            "multiselect": True,
            "navigationButtons": True
          },
          "physics": {
            "minVelocity": 0.75
          }
        } 
        cluster_net.options = const_options
        #cluster_net.toggle_physics = False
        #label_dict = dict(zip(port_data['ISIN'],port_data['ticker']))
        
        for node in cluster_net.nodes:
            
            node["group"] = inv_map[node['id']]
            #if node['group'] == -1:
                #node['label'] = ''
            #else:
            node["label"] = label_dict[node['label']]
            
            #node["title"] = 'Asset {0} \n Weight {1}% \n Cluster {2} '.format(name_dict[node['id']],np.round((weight_dict[node['id']]*100),2),node["group"])
            node['size'] = 10

        #cluster_net.edges = [x for x in cluster_net.edges if x['corr'] >= thresh]
        for edge in cluster_net.edges:
                edge_from = label_dict[edge['from']]
                edge_to = label_dict[edge['to']]
                weight = np.round(edge['corr'],3)
                edge_title = "ρ({0},{1}) = {2}".format(edge_from,edge_to,str(weight))
                edge["title"] = edge_title
                edge['width'] = weight*3
                
        return cluster_net
    
    def create_color_palette(self,color_map,scaled=False,as_hex=False):
        keys = list(set(color_map.values()))
        palette = sns.color_palette("husl", len(keys))
        
        color_palette = dict(zip(keys,palette))
        
        if -1 in keys:
            color_palette[-1] = (0.82,0.82,0.82)

        if scaled:
            
            palette = ['rgb'+str(tuple(x*255)) for x in np.array(list(color_palette.values()))]
            color_palette = dict(zip(keys,palette))
            
        if as_hex: 

            color_palette = dict(zip(keys,palette.as_hex()))
            color_palette[-1] = '#808080'

        return color_palette


    def color_network(self,_cluster_net,color_map):
        
        #if color_map_choice is 'Cluster':
            #f = create_legend(color_map=None)
            #return cluster_net,f
        #else:
        color_palette = self.create_color_palette(color_map,scaled=True)
        
        for node in _cluster_net.nodes:
            node['group'] = color_map[node['id']]
            node['color'] = color_palette[color_map[node['id']]]

        f = self.create_legend(color_map)
        
        self.color_palette = color_palette
        
        return _cluster_net,f
    
        
    def create_legend(self,color_map,figsize=(10,3)):
        
        f,ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        f.set_facecolor('none')
        f.patch.set_alpha(0)
    
        if color_map is not None:
            color_palette = self.create_color_palette(color_map,scaled=False)
            
            for k,v in color_palette.items():
                
                if type(k) == int:
                    if k == -1:
                        label = 'Unclustered Assets'
                    else:
                        label = 'Cluster {}'.format(k)
                else:
                    label = k
                
                ax.plot([0],[0],linewidth=4.0,label=label,color=v)#color=eval(v.replace('rgb','')),label=k)
                
    
            f.legend(loc='lower center', frameon=False,fontsize='16')
    
            #f.tight_layout()
            
        return f


    def visualize_cluster_evolution(self,cluster_assignments,label_dict=None,height=1000,width=1500):
        
        def remap_col(col):
            m = dict(zip(col.unique(),range(len(col))))
            return col.map(m)
        
        def get_asset_look_up(df_links):
            inv = df_links.stack().reset_index()
            inv.columns = ['asset','date','cluster']
            inv = inv.groupby(['date','cluster'])['asset'].apply(list).reset_index()
            asset_look_up = inv.pivot(index='cluster',columns='date',values='asset')
            asset_look_up = asset_look_up.stack().reset_index()
            asset_look_up = asset_look_up.rename(columns={0:'asset'})
            asset_look_up = dict(zip(asset_look_up.index,asset_look_up.asset))
            return asset_look_up
        
        def rgb_to_hex(rgbstr):
            rgb = tuple([int(float(x)) for x in re.sub("[^0-9.,]","",rgbstr).split(',')])
            return '#%02x%02x%02x' % rgb
        
        def hex_to_rgb(hexstr,opacity=None):
            hexstr = hexstr.replace('#','')
            
            if opacity:
                tup = tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))+(opacity,)
                return 'rgba'+str(tup)
            else:
         
                tup = tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))
                return 'rgb'+str(tup)
            
        def check_palette(palette):
            if any(['rgb' in x for x in palette.values()]):
                return {k:rgb_to_hex(v) for k,v in palette.items()}
            else:
                return palette
            
        def factorize(s):
            a = pd.factorize(s, sort=True)[0]
            return (a + 0.01) / (max(a) + 0.1)

        def jaccard_similarity(list1, list2):
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(set(list1)) + len(set(list2))) - intersection
            return float(intersection) / union
        
        def cluster_similarity(x,asset_list,cutoff=0.3):
            
            sim = [jaccard_similarity(x,y) for y in asset_list]
            
            return np.argmax(sim) if any([x>cutoff for x in sim]) else np.nan
        
        def group_nodes(node_df,cutoff=0.5):
            node_df['time_key']=node_df['time_key'].astype(int)
            time_steps = sorted(list(set(node_df['time_key'])))
            terminal = node_df['time_key'] == max(time_steps)
           # num_terminal_nodes = len(node_df[terminal])
            node_df.loc[terminal,'color_key'] = node_df.loc[terminal,'cluster_key']
            
            if any(node_df['cluster_label'] == 'Unclustered Assets'):
                offset = 1
            else:
                offset = 0
            
            node_df.loc[node_df['cluster_label'] == 'Unclustered Assets','color_key'] = 0
            for i in reversed(range(1,len(time_steps)+1)):
                j = i-1
                
                t = node_df.loc[(node_df['time_key'] == i) & (node_df['cluster_label'] != 'Unclustered Assets')]
                s = node_df.loc[(node_df['time_key'] == j) & (node_df['cluster_label'] != 'Unclustered Assets')]
   
                s_key = s['asset'].apply(lambda x: cluster_similarity(x,list(t['asset']),cutoff=cutoff))
                s_key = s_key.map(dict(zip(t['cluster_key']-offset,t['color_key'])))
                
                no_match = s_key[s_key.isna()]
                if len(no_match) >0:
                    #max_k = int(max(s_key.dropna())+1)
                    max_k = int(max(node_df['color_key'].dropna()))+1
            
                    nan_key = list(range(max_k,max_k+len(s_key[s_key.isna()])))
    
                    s_key.loc[s_key.isna()] = nan_key
                    
                node_df.loc[s_key.index,'color_key'] = s_key
            return node_df

           # tmp['color_key2'] = tmp['color_key2'].fillna(tmp['cluster_key'])
        #cluster_series = cluster_assignments.set_index('date')
        #mask = cluster_assignments.index.isin(cluster_assignments.index[::time_step]) | (cluster_assignments.index == cluster_assignments.index[-1])
        #cluster_series=cluster_assignments.loc[mask]
        cluster_series = cluster_assignments.copy()
        if label_dict is None:
           label_dict = self.map_tickers(cluster_series.columns)
        
        cluster_series.columns.name = 'asset'
        cluster_series.columns = cluster_series.columns.map(label_dict)
        
        if any((cluster_series == -1).any()):
            cluster_series = cluster_series+1
            zero_unclustered = True
        else:
            zero_unclustered = False
            
        df_links = cluster_series.T
        df_links = df_links[~df_links.index.duplicated(keep=False)]
        sort_idx = -1
        
        largest_cluster = df_links.iloc[:,sort_idx].value_counts()
        largest_cluster =df_links.melt()['value'].value_counts()
        reidx = df_links.iloc[:,sort_idx].map(dict(zip(largest_cluster.index,range(len(largest_cluster))))).sort_values()
        df_links = df_links.reindex(reidx.index)

        #df_links = df_links.sort_values(by=df_links.columns[sort_idx])
        #df_links = df_links.apply(lambda x: remap_col(x))

        asset_look_up = get_asset_look_up(df_links)
        df_links = df_links.reset_index(names=['asset'])
        time_pts = list(cluster_series.index)

        groups = df_links.groupby(time_pts).agg({'asset':'count'})
        #assets=df_links.groupby(list(gr.index))['asset'].apply(list)
        _iterator = iter(range(1,len(time_pts)))
        list_ = []
        for idx in _iterator:
            jidx = idx-1
            combos = groups.groupby([time_pts[jidx],time_pts[idx]]).agg({'asset':'sum'}).reset_index()
            #combos = combos[(combos>=0).all(axis=1)]
            list_.append(combos)

        count_dict = {}

        for i in range(0, len(list_)): 
           cols =list_[i].columns # contains columns for our dataframe 
        #(list_[i]) 
         #This for loop is inside the outer loop
           for x,y,z in zip(list_[i][cols[0]],list_[i][cols[1]],list_[i][cols[2]]):#Iterates over x(source),y(target),z(counts)
               x = str(x)
               y = str(y)
               count_dict[x+'_M'+str(i+1),y+'_M'+str(i+2)] = z
                    
        df = pd.DataFrame.from_dict(count_dict,orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index, names=['source', 'target'])
        df = df.reset_index()
        df.columns = ['source','target','value']
        df = df.loc[df["source"].str[-1].apply(ord) < df["target"].str[-1].apply(ord)]
        df = df.groupby(["source", "target"], as_index=False).sum()
        #df['sim']=df.apply(lambda x: jaccard_similarity(x['source_assets'],x['target_assets']),axis=1)
        
        #cluster_num = max(df['value'])+1
        color_list = af.color_list(50)
        if hasattr(self,'color_palette'):
            
            self.color_palette = check_palette(self.color_palette)
            self.color_palette = {k:v for k,v in self.color_palette.items() if k != -1}
            color_list = [x for x in color_list if x not in list(self.color_palette.values())]
            for i,x in enumerate(range(max(self.color_palette.keys())+1,len(color_list))):
                self.color_palette[x] = color_list[i]

        else:
            color_map = dict(zip(range(len(color_list)),color_list))

            self.color_palette = color_map
        

        opacity = 0.3
        color_palette_o = [hex_to_rgb(v) for v in self.color_palette.values()]
#        color_palette_o.extend(['rgb(211,211,211)'])
        color_palette_t = [hex_to_rgb(v,opacity=opacity) for v in self.color_palette.values()]
        #color_palette_t.extend(['rgba(211,211,211,0.3)'])


        # unique nodes
        nodes = np.unique(df[["source", "target"]], axis=None)
        nodes = pd.Series(index=nodes, data=range(len(nodes)))
        nodes = (
            nodes.to_frame("id")
            .assign(
                x=lambda d: factorize(d.index.str[-1]),
                y=lambda d: factorize(d.index.str[:-1]),
            )
        )
        #df['source_key'] = df['source'].apply(lambda x: x.split('_M')[0])
        #hoverinfo = hoverinfo.apply(lambda x: '<br>'.join(y for y in x))
        nodes['asset'] = nodes['id'].map(asset_look_up)
        nodes['asset_label'] = nodes['asset'].apply(lambda x: '<br>'.join(x))
        
        nodes['cluster_key'] = [int(x.split('_M')[0]) for x in nodes.index]
        nodes['time_key'] = [int(x.split('_M')[1]) for x in nodes.index]
        nodes['cluster_label'] = ['Cluster {}'.format(x.split('_M')[0]) for x in nodes.index]

        if zero_unclustered:
            
            zero_mask = nodes['cluster_key'] == 0
            #nodes.loc[~zero_mask,'cluster_key'] = nodes.loc[zero_mask]
            nodes['cluster_label'] = ['Cluster {}'.format(x-1) for x in nodes.cluster_key]
            
            nodes.loc[zero_mask,'cluster_label'] = 'Unclustered Assets'

            color_palette_o = ['rgb(255,255,255)'] + color_palette_o
            color_palette_t = ['rgba(211,211,211,{})'.format(opacity)] + color_palette_t
            

        nodes = group_nodes(nodes,cutoff=0.5)
        nodes['color'] = [color_palette_o[int(x)] for x in nodes.color_key]
        
        if zero_unclustered:
            nodes.loc[zero_mask,'color'] = 'rgb(255,255,255)'
            nodes.loc[zero_mask,'color_key'] = '0'
        
        df['source_assets'] = df['source'].map(dict(zip(nodes.index,nodes['id']))).map(asset_look_up)
        df['target_assets'] = df['target'].map(dict(zip(nodes.index,nodes['id']))).map(asset_look_up)
       

        #d#df = df.sort_values(by=['target'])
        df['flow'] = df.apply(lambda row: [x for x in row['source_assets'] if x in row['target_assets']],axis=1)
        df['flow'] = df['flow'].apply(lambda x: '<br>'.join(x))
        
        df['color'] = [color_palette_t[int(x)] for x in nodes.loc[df["source"]].color_key]
        
        
        
        df['time_key'] = [x.split('_M')[1] for x in df.source]
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node={"label": nodes['cluster_label'], 
                      "customdata":nodes['asset_label'],
                      "hovertemplate":'%{customdata}',
                      "x": nodes["x"], 
                      "y": nodes["y"],
                      'color':nodes['color']},
                      #"color": [color_palette_o[int(x.split('_M')[0])] for x in nodes.index]},
                link={
                    "source": nodes.loc[df["source"], "id"],
                    "target": nodes.loc[df["target"], "id"],
                    "value": df["value"],
                    "label":df['flow'],
                    "color":df['color'],
                   # "hovertemplate":'From %{source.customdata} to %{target.customdata} <br> %{customdata}',
                   # "color": [color_palette_t[int(x.split('_M')[0])] for x in nodes.loc[df["source"]].index]
                },
            )
        )

        time_pt_disp = [x.strftime('%m/%d/%Y') for x in time_pts]

        for i in range(0,len(time_pt_disp)):
            fig.add_annotation(
                  x=i,#Plotly recognizes 0-5 to be the x range.
                  y=1.1,#y value above 1 means above all nodes
                  xref="x",
                  yref="paper",
                  text=time_pt_disp[i],#Text
                  showarrow=False,
                  font=dict(
                      family="Tahoma",
                      size=16,
                      color="black"
                      ),
                  align="left",
                  )
            
        fig.update_layout(go.Layout(width=width,height=height,                                
                                        xaxis =  {                                     
                                            'showgrid': False,
                                            'zeroline': False, 
                                            'visible':False
                                                 },
                                        yaxis = {                              
                                            'showgrid': False,
                                            'zeroline': False, 
                                            'visible':False
                                                }), plot_bgcolor='white')

        #fig.write_html('history.html',auto_open=True)
        return fig

    
    def save_network(self,cluster_net,name='corr.html'):
        cluster_net.save_graph(name)
        
