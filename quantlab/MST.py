#%%
import pandas as pd
import numpy as np
import networkx as nx
import datetime as dt
import matplotlib.pyplot as plt
import time
from datetime import datetime , timedelta
from statsmodels import regression
import statsmodels.api as sm
import csv
from IPython.display import clear_output
import pandas_datareader as pdr

class MinimumSpanningTree(object):
    """
    Using data as an input the MST will be created
    Attributes:
    data: Data is retrived from bloomberg
    """
    def __init__(self,data,period):
        """Initialising the attributes"""
        self.data=data
        self.period=period
    
    def log_returns(self):
        """Log returns are calculated an the NA column is removed"""
        log_returns=np.log(self.data.pct_change(1)+1).dropna()
        return log_returns

    def corr(self):
        """Using log return the covariance matrix is calculated for a period"""
        log_returns = self.log_returns()[-self.period:]
        return log_returns.corr()

    def dist_matrix(self):
        """This distance matrix is used"""
        return np.sqrt(2*(1-self.corr()))

    def MST_matrix(self):
        """Minimum Spanning Tree is created"""
        distm = self.dist_matrix()
        X = csr_matrix(distm.values)
        Tcsr = minimum_spanning_tree(X).toarray()
        mst_df = pd.DataFrame(Tcsr,index=distm.columns,columns=distm.columns)
        return mst_df

    def MST_DF_NX(self,reduce_labels=True):
        """The MST Matrix is called"""
        MST_matrix = self.MST_matrix()
        """The ticker names are reduced in order to view the graphs clearly.
         This includes removing the exchange and the asset class e.g. UN Equity"""
        if reduce_labels == True:
            MST_matrix.columns = [name.split()[0] for name in MST_matrix.columns]
            MST_matrix.index = MST_matrix.columns
        """The data is reformatted into a pandas dataframe with the following format:
        Node1, Node2, Weight"""
        mst=[]
        for name in MST_matrix.index:
            for name1 in MST_matrix.columns:
                if MST_matrix[name][name1]>0:
                    mst.append([name,name1,MST_matrix[name][name1]])
        mst = pd.DataFrame(mst)
        mst.columns = ['node1','node2','weight']
        return mst

    def create_graph(self):
        mst_df = self.MST_DF_NX()
        G = nx.from_pandas_dataframe(mst_df, 'node1', 'node2', ['weight'])
        return G

    def factors(self, as_frame = False):
        G = self.create_graph()
        factors = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'eccentricity']

        factor_dict = {}

        factor_dict['degree'] = nx.degree_centrality(G)
        factor_dict['betweenness'] = nx.betweenness_centrality(G)
        factor_dict['closeness'] = nx.closeness_centrality(G)
        factor_dict['eigenvector'] = nx.eigenvector_centrality(G)
        factor_dict['pagerank'] = nx.pagerank_numpy(G)
        factor_dict['eccentricity'] = nx.eccentricity(G)
        
        if as_frame:
            return pd.DataFrame.from_dict(factor_dict)
        else:
            return factor_dict
    