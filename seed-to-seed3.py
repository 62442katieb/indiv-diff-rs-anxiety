from __future__ import division
import numpy as np
from glob import glob
from os.path import join, basename, exists
from os import makedirs
import matplotlib.pyplot as plt
from nilearn import input_data
from nilearn import datasets
import pandas as pd
from nilearn import plotting
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import bct
from matplotlib import pyplot as plt

def betweenness_wei(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.
    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix
    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector
    Notes
    -----
       The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix.
       Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC

laird_2011_icns = '/home/data/nbc/SeedToSeed/17-networks-combo-ccn-5.14.nii.gz'

network_masker = input_data.NiftiLabelsMasker(laird_2011_icns, standardize=True)
#region_masker = input_data.NiftiLabelsMasker('/scratch/kbott/salience-anxiety/relabeled_yeo_atlas7.nii.gz', standardize=True)

subjects = ['101', '102', '103', '104', '106', '107', '108', '110', '212', '213',
            '214', '215', '216', '217', '218', '219', '320', '321', '322', '323',
            '324', '325', '327', '328', '329', '330', '331', '332', '333', '334',
            '335', '336', '337', '338', '339', '340', '341', '342', '343', '344',
            '345', '346', '347', '348', '349', '350', '451', '452', '453', '455',
            '456', '457', '458', '459', '460', '462', '463', '464', '465', '467',
            '468', '469', '470', '502', '503', '571', '572', '573', '574', '575',
            '577', '578', '579', '580', '581', '582', '584', '585', '586', '587',
            '588', '589', '590', '591', '592', '593', '594', '595', '596', '597',
            '598', '604', '605', '606', '607', '608', '609', '610', '611', '612',
            '613', '614', '615', '616', '617', '618', '619', '620', '621', '622',
            '623', '624', '625', '626', '627', '628', '629', '630', '631', '633',
            '634']
#subjects = ['101']

data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
work_dir = '/home/data/nbc/SeedToSeed'
sink_dir = '/home/kbott006/salience'
#data_dir = '/Users/Katie/Dropbox/Data'
#work_dir = '/Users/Katie/Dropbox/Data/salience-anxiety-graph-theory'

for s in subjects:
    if not exists(join(sink_dir, s)):
        makedirs(join(sink_dir, s))
    fmri_file = join(work_dir, '{0}_filtered_func_data_mni.nii.gz'.format(s))

    motion = np.genfromtxt(join(data_dir, s, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'mc', 'prefiltered_func_data_mcf.par'))
    outliers_censored = join(work_dir, '{0}_confounds.txt'.format(s))
    if exists(outliers_censored):
        print "outliers file exists!"
        outliers = np.genfromtxt(outliers_censored)
        #cat = np.hstack((motion, outliers))
        #np.savetxt((join(sink_dir, s, '{0}_confounds.txt'.format(s))), cat)
        confounds = outliers_censored

    else:
        print "No outliers file found for {0}".format(s)
        np.savetxt((join(sink_dir, s, '{0}_confounds.txt'.format(s))), motion)
        confounds = join(data_dir, s, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'mc', 'prefiltered_func_data_mcf.par')

    network_time_series = network_masker.fit_transform (fmri_file, confounds)
    #region_time_series = region_masker.fit_transform (fmri_file, confounds)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    network_correlation_matrix = correlation_measure.fit_transform([network_time_series])[0]
    np.savetxt(join(sink_dir, s, '{0}_network_corrmat_Laird2011.csv'.format(s)), network_correlation_matrix, delimiter=",")

    network = {}
    #region = {}
    network_wise = {}
    #region_wise = {}

    #talking with Kim:
    #start threhsolding (least conservative) at the lowest threshold where you lose your negative connection weights
    #steps of 5 or 10 percent
    #citation for integrating over the range is likely in the Fundamentals of Brain Network Analysis book
    #(http://www.danisbassett.com/uploads/1/1/8/5/11852336/network_analysis_i__ii.pdf)
    #typically done: make sure your metric's value is stable across your range of thresholds
    #the more metrics you use, the more you have to correct for multiple comparisons
    #make sure this is hypothesis-driven and not fishing

    for p in np.arange(0.1, 1, 0.1):
        ntwk = []
        ntwk_wise = []

        ntwk_corrmat_thresh = bct.threshold_proportional(network_correlation_matrix, p, copy=True)
        #network measures of interest here

        #global efficiency
        le = bct.efficiency_wei(ntwk_corrmat_thresh)
        ntwk.append(le)

        #clustering coefficient
        c = bct.clustering_coef_wu(ntwk_corrmat_thresh)
        ntwk_wise.append(c)

        network[p] = ntwk
        network_wise[p] = ntwk_wise

    ntwk_df = pd.DataFrame(network).T
    ntwk_df.columns = ['total positive', 'total negative', 'efficiency', 'path length', 'modularity']

    ntwk_wise_df = pd.DataFrame(network_wise).T
    ntwk_wise_df.columns = ['betweenness', 'degree', 'positive weights', 'negative weights',
                                                       'community index', 'clustering coefficient']
    ntwk_df.to_csv(join(sink_dir, s, '{0}_network_metrics.csv'.format(s)), sep=',')
    ntwk_wise_df.to_csv(join(sink_dir, s, '{0}_network_wise_metrics.csv'.format(s)), sep=',')
