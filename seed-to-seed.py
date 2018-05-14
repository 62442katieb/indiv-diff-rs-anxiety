from __future__ import division
import numpy as np
from glob import glob
from os.path import join, basename, exists
import matplotlib.pyplot as plt
from nilearn import input_data
from nilearn import datasets
import nipype
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

#Yeo_atlas = datasets.fetch_atlas_yeo_2011(data_dir='/home/kbott006/nilearn_data', url=None, resume=True, verbose=1)

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.thick_17

#region_labels = regions.connected_label_regions(atlas_yeo)
#region_labels.to_filename('/scratch/kbott/salience-anxiety/relabeled_yeo_atlas.nii.gz')

network_masker = input_data.NiftiLabelsMasker(atlas_yeo, standardize=True)
region_masker = input_data.NiftiLabelsMasker('/scratch/kbott/salience-anxiety/relabeled_yeo_atlas.nii.gz', standardize=True)

subjects = ['101']
#subjects = ['101', '102', '103', '104', '106', '107', '108', '110', '212', '213',
            #'214', '215', '216', '217', '218', '219', '320', '321', '322', '323',
            #'324', '325', '327', '328', '329', '330', '331', '332', '333', '334',
            #'335', '336', '337', '338', '339', '340', '341', '342', '343', '344',
            #'345', '346', '347', '348', '349', '350', '451', '452', '453', '455',
            #'456', '457', '458', '459', '460', '462', '463', '464', '465', '467',
            #'468', '469', '470', '502', '503', '571', '572', '573', '574', '575',
            #'577', '578', '579', '580', '581', '582', '584', '585', '586', '587',
            #'588', '589', '590', '591', '592', '593', '594', '595', '596', '597',
            #'598', '604', '605', '606', '607', '608', '609', '610', '611', '612',
            #'613', '614', '615', '616', '617', '618', '619', '620', '621', '622',
            #'623', '624', '625', '626', '627', '628', '629', '630', '631', '633',
            #'634']

data_dir = '/home/ariegonz/SeedToSeed'
work_dir = '/scratch/kbott/salience-anxiety'
physics = '/home/data/nbc/physics-learning/data/pre-processed'
#data_dir = '/Users/Katie/Dropbox/Data'
#work_dir = '/Users/Katie/Dropbox/Data/salience-anxiety-graph-theory'

for s in subjects:
    fmri_file = join(data_dir, '{0}_filtered_func_data_mni.nii.gz'.format(s))
    motion = np.genfromtxt(join(physics, s, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'mc', 'prefiltered_func_data_mcf.par'))
    outliers = np.genfromtxt(join(physics, s, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'motion_outliers-censored.txt'))
    cat = np.hstack((motion, outliers))
    np.savetxt((join(data_dir, '{0}_confounds.txt'.format(s))), cat)
    confounds = join(data_dir, '{0}_confounds.txt'.format(s))
    network_time_series = network_masker.fit_transform (fmri_file, confounds)
    region_time_series = region_masker.fit_transform (fmri_file, confounds)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    #labels = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB', 'ContC', 'ContA', 'ContB', 'DefaultD', 'DefaultC', 'DefaultA', 'DefaultB']
    network_correlation_matrix = correlation_measure.fit_transform([network_time_series])[0]
    region_correlation_matrix = correlation_measure.fit_transform([region_time_series])[0]
    np.savetxt('region_corrmat_Yeo17.csv', region_correlation_matrix, delimiter=",")
    #labeled_netwk_corrmat = np.hstack((labels, network_correlation_matrix))
    #labels_plus = [' ', 'VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB', 'ContC', 'ContA', 'ContB', 'DefaultD', 'DefaultC', 'DefaultA', 'DefaultB']
    #labeled_netwk_corrmat = np.vstack((labels_plus, labeled_netwk_corrmat))
    np.savetxt('network_corrmat_Yeo17.csv', network_correlation_matrix, delimiter=",")

    L = []
    BC = []
    DEG = []
    SPOS = []
    SNEG = []
    VPOS = []
    VNEG = []
    LE = []
    CI = []
    Q = []
    PL = []
    GL_EFF = []
    ECC = []
    RAD = []
    DIA = []
    MOD = []
    C = []

    for p in np.arange(0.1, 1, 0.1):
        ntwk_corrmat_thresh = bct.threshold_proportional(network_correlation_matrix, p, copy=True)
        regn_corrmat_thresh = bct.threshold_proportional(region_correlation_matrix, p, copy=True)
        #network measures of interest here
        #betweenness centrality
        l = bct.weight_conversion(regn_corrmat_thresh, 'lengths')
        bc = betweenness_wei(l)
        BC.append(bc)

        #node degere
        deg = bct.degrees_und(regn_corrmat_thresh)
        DEG.append(deg)

        #node strength
        [Spos, Sneg, vpos, vneg] = bct.strengths_und_sign(regn_corrmat_thresh)
        #nodal strength of positive weights
        SPOS.append(Spos)
        #nodal strength of negative weights
        SNEG.append(Sneg)
        #total positive weight
        VPOS.append(vpos)
        #total negative weight
        VNEG.append(vneg)

        #nodal efficiency
        le = bct.efficiency_wei(regn_corrmat_thresh)
        LE.append(le)

        #path length
        [pl, gl_eff, ecc, radius, diameter] = bct.charpath(regn_corrmat_thresh)
        PL.append(pl)
        GL_EFF.append(gl_eff)
        ECC.append(ecc)
        RAD.append(radius)
        DIA.append(diameter)

        #modularity (for non-overlapping community structure)
        [ci, q] = bct.modularity_louvain_und(regn_corrmat_thresh, gamma=1, hierarchy=False)
        #modules = bct.ci2ls(ci)
        #modules = np.asarray(modules)
        MOD.append(ci)
        Q.append(q)

        #clustering coefficient
        c = bct.clustering_coef_wu(regn_corrmat_thresh)
        C.append(c)

        #salience network, default mode, executive control
    #where rows are proportional thresholds 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    np.savetxt((join(work_dir, '{0}_betweenness_centrality.csv'.format(s))), BC, delimiter=',')
    np.savetxt((join(work_dir, '{0}_node_degree.csv'.format(s))), DEG, delimiter=',')
    np.savetxt((join(work_dir, '{0}_positivenode_strength.csv'.format(s))), SPOS, delimiter=',')
    np.savetxt((join(work_dir, '{0}_negativenode_strength.csv'.format(s))), SNEG, delimiter=',')
    np.savetxt((join(work_dir, '{0}_total_positive_weight.csv'.format(s))), VPOS, delimiter=',')
    np.savetxt((join(work_dir, '{0}_total_negative_weight.csv'.format(s))), VNEG, delimiter=',')
    np.savetxt((join(work_dir, '{0}_nodal_efficiency.csv'.format(s))), LE, delimiter=',')
    np.savetxt((join(work_dir, '{0}_path_length.csv'.format(s))), PL, delimiter=',')
    np.savetxt((join(work_dir, '{0}_global_efficiency.csv'.format(s))), GL_EFF, delimiter=',')
    np.savetxt((join(work_dir, '{0}_eccentricity.csv'.format(s))), ECC, delimiter=',')
    np.savetxt((join(work_dir, '{0}_radius.csv'.format(s))), RAD, delimiter=',')
    np.savetxt((join(work_dir, '{0}_diameter.csv'.format(s))), DIA, delimiter=',')
    np.savetxt((join(work_dir, '{0}_community_index.csv'.format(s))), MOD, delimiter=',')
    np.savetxt((join(work_dir, '{0}_modularity.csv'.format(s))), Q, delimiter=',')
    np.savetxt((join(work_dir, '{0}_clustering_coefficient.csv'.format(s))), C, delimiter=',')

    # Plot the correlation matrix

    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:
    np.fill_diagonal(ntwk_corrmat_thresh, 0)

    plt.imshow(ntwk_corrmat_thresh, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

    # Add labels and adjust margins
    x_ticks = plt.xticks(range(len(labels)), labels[:], rotation=90)
    y_ticks = plt.yticks(range(len(labels)), labels[:])
    plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
    plt.savefig('{0}_correlation_graph.png'.format(s), dpi=400)
