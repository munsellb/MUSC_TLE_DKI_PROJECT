import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import image # for resampling of wm mask
from scipy.stats.mstats import zscore
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score


def load_db(base_dir,classes):
    """ create a pandas DataFrame based on Parameters
    
    Parameters
    ----------
    base_dir : string
        directory structured with each folder representing mode and each mode folder 
        containing nifti volumes.
    classes : list of tuples
        tuple of form [(row label for Dataframe,file name Identifier)...].
    
    Returns
    ----------
    df : pandas.DataFrame
        DataFrame of shape (Classes,Modes) contain a list of file paths at each index.
    
    """
    dir_list = sorted(os.listdir(base_dir))
    df = pd.DataFrame(columns=[i[0] for i in classes],index = dir_list )
    for directory in dir_list:
        for c in classes:
            path = '{}{}{}'.format(base_dir,directory,'/')
            df.set_value(directory,c[0],[path + d for d in os.listdir(path) if c[1] in d])
    return df
def create_mask(wm,epi,threshold =.5):
    """Resample a brain mask to an epi
    
    Parameters
    ----------
    wm : nibabel.Nifti1Image 
        Mask as Nibabel Nifti1Image Object 
        
    epi : nibabel.Nifti1Image 
        EPI as Nibabel Nifti1Image Object
        
    threshold : float, between 0 and 1 (default=.5)
        Threshold at which to apply mask to epi
        
    Returns
    ----------
    mask : nibabel.Nifti1Image
        Resampled brain mask
    """
    mask = image.resample_img(wm,target_shape=epi.shape, target_affine=epi.get_affine())
    img = mask.get_data()
    img[img > threshold] = 1.
    img[img <= threshold] = 0.
    return nib.Nifti1Image(img,np.eye(4))

def apply_mask_to_epi(mask_path,epi_path,threshold=.5):
    """Apply a mask to an epi and return a numpy array of 1xN_VOXELS
    
    Parameters
    ----------
    mask_path : String
        file path of mask nifti volume
        
    epi_path : String
        file path of epi nifti volume
        
    threshold : float, between 0 and 1 (default=.5)
        Threshold at which to apply mask to epi
        
    Returns
    ----------
    masked_vol : numpy.ndarray
    an array of voxels corresponding to mask
    """
    mask_data = create_mask(nib.load(mask_path),nib.load(epi_path)).get_data()
    epi_data = nib.load(epi_path).get_data()
    mask_data = mask_data.reshape((-1,1))
    epi_data = epi_data.reshape((-1,1))
    epi_data = epi_data[mask_data ==1]
    return epi_data
def list_mask_apply(l,mask_path,threshold=.5):
    """ helper function for mask_dataframe"""
    ret = []
    for vol in l:
        ret.append(apply_mask_to_epi(mask_path,vol,threshold))
    return ret

def mask_dataframe(df,mask_path,threshold=.5):
    """Apply a mask to an epi and return a numpy array of 1xN_VOXELS
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame object to mask
  
    mask_path : string
        file path of mask nifti volume

    threshold : float, between 0 and 1 (default=.5)
        Threshold at which to apply mask to epi
        
    Returns
    ----------
    df : pandas.DataFrame
        Dataframe with lists of file names changed to lists of masked ndarrays
    """
    for mode, i in df.iterrows():
        for i,l in enumerate(df.ix[mode]):
            df.ix[mode][i] = list_mask_apply(l,mask_path,threshold)
    return df




# MAKE SURE THIS IS RIGHT

def df_scores(grid_search,gamma_range,C_range):    
    """Create a dataframe of scores of params
    
    Parameters
    ----------
    grid_search : sklearn.grid_search.GridSearchCV
        grid search object after it has been fit
  
    gamma_range : numpy.ndarray
        range of gamma values used to fit grid
        
    C_range : numpy.ndarray
        range of C values used to fit grid
        
    Returns
    ----------
    df : pandas.DataFrame
        labeled dataframe showing scores 
        
    scores : numpy.ndarray
        array of scores
    """
    scores = np.array([x[1] for x in grid_search.grid_scores_]).reshape(len(C_range), len(gamma_range))
    return pd.DataFrame(scores,index=C_range,columns=gamma_range)



def plot_scores(grid_search,gamma_range,C_range):
    """Plot scores with matploblib imshow from a grid search
    
    Parameters
    ----------
    grid_search : sklearn.grid_search.GridSearchCV
        grid search object after it has been fit
  
    gamma_range : numpy.ndarray
        range of gamma values used to fit grid
    C_range : numpy.ndarray
        range of C values used to fit grid
        
    """
    plt.figure(figsize=(15,10))
    scores = np.array([x[1] for x in grid_search.grid_scores_]).reshape(len(C_range), len(gamma_range))
    cax = plt.imshow(scores, interpolation="nearest", cmap=plt.cm.cool)
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.xlabel('$\gamma$')
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.ylabel('C')
    plt.title('Validation accuracy')
    cbar = plt.colorbar(cax, ticks=[np.min(scores), np.mean(scores), np.max(scores)])
    plt.show()
    print 'Max Performance : {}'.format((np.round(np.max(scores),5)))
    


def build_matrix(masked_matrices,modes):
    """build subject x feature matrix based on modalities
    
    Parameters
    ----------
    masked_matrices : pandas.DataFrame
        dataframe of lists
  
    modes : list
        list of modalities to include

    Returns
    ----------
    x_input : np.array
    subject x feature matrix
        
    """
    if len(modes) == 1:
        x_input = np.vstack((masked_matrices.ix[modes[0]].cons,masked_matrices.ix[modes[0]].pats))
    else :
        inputs = []
        for m in modes:
            inputs.append( np.vstack((masked_matrices.ix[m].cons,masked_matrices.ix[m].pats)))
        x_input = np.hstack(inputs)
    return x_input



def zscore_mag_div_matrix(mat):
    for col in xrange(mat.shape[1]):
        if col % 50000 == 0:
            print col
        mat[:,col] = zscore(mat[:,col]) 
        mat[:,col] = mat[:,col] / np.sqrt( sum( [ x**2 for x in mat[:,col] ] ) )
    return mat

def build_y(masked_matrices,mode):
    y_len = map(len,masked_matrices.ix[0])
    n_samples = np.sum(y_len)
    y_labels_vector = np.vstack((np.array([0] * y_len[0]).reshape((-1,1)),np.array([1] * y_len[1]).reshape((-1,1)))).reshape(n_samples)
    print 'Y label vector is {} elements, with {} False and {} True\n'.format(len(y_labels_vector),y_len[0],y_len[1])
    print 'y_labels_vector.shape : {}       ( Should be (n_samples,) )'.format(y_labels_vector.shape)
    return y_labels_vector

def MakeBinary(w):
    w[w > 0] = 1
    w[w <= 0 ] = 0
    return w


def imshow_data(data_frame):
    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    rcParams['xtick.labelsize'] = 12
    rcParams['axes.titlesize']=  30
    rcParams['axes.labelsize'] = 30
    alp = np.unique(data_frame.alpha)
    gam = np.unique(data_frame.gamma)
    scores = []
    nopes = []
    fig = plt.figure(figsize=(18,12))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    anots = []
    
    for i,a in enumerate(alp):
        d = data_frame[data_frame.alpha == a]
        for j,g in enumerate(gam):
            try:
                scores.append(np.float(d[d.gamma == g]['results_mean']))
            except:
                scores.append(-1)
                nopes.append((a,g))
    features = data_frame.sort('alpha').drop_duplicates('alpha').n_features
                
    scores = np.array(scores)
    scores = np.array(scores.reshape((len(alp),len(gam))))
    cax = ax.imshow(scores, interpolation="nearest", cmap=plt.cm.bone)
    for i in nopes:
        print i
    plt.xticks(np.arange(len(gam)),gam, rotation=65)
    plt.xlabel(r'$\gamma$')
    plt.yticks(np.arange(len(alp)),['{1:>3}   {0:>10}'.format('({})'.format(int(features.iloc[i])), alp[i]) for i in xrange(len(alp))])
    plt.ylabel(r'$\alpha\;(features)$')
    plt.title('Validation accuracy')
    cbar = plt.colorbar(cax, ticks=[np.min(scores), np.mean(scores), np.max(scores)])
    print 'Max Performance : {}'.format((np.round(np.max(scores),5)))
    
    for i,a in enumerate(alp):
        d = data_frame[data_frame.alpha == a]
        for j,g in enumerate(gam):
            try:
                sc_ = np.float(d[d.gamma == g]['results_mean'])
            except:
                sc_ = -1
            if np.max(scores) - sc_ < .05:
                w = 'normal'
                if sc_ == np.max(scores):
                    w = 'semibold'
                ax.annotate(str(np.round(sc_,2))[1:], xy=(j, i),weight=w, horizontalalignment='center', verticalalignment='center')

def lasso_feature_selection(x,y,alpha):
    ''' runs feature selection on with Lasso '''
    en = Lasso(alpha=alpha,
           fit_intercept=False,
           normalize=False,
           copy_X = True,
           max_iter=10000
          )
    
    en.fit(x,y.astype(np.float))
    w = MakeBinary(en.coef_)
    print 'Weights Shape : {}'.format(w.shape)
    x *= w
    print '{} positive coefficients @ alpha = {}'.format(np.count_nonzero(w),alpha)
    print '{} positive coefficients @ alpha = {}'.format(np.count_nonzero(x)/len(y),alpha)
    return x , w




def coefs_mask(coefs,resampled_mask):
    l_coefs = list(coefs)
    rav_mask = resampled_mask.get_data().ravel()
    z_mask = np.zeros(rav_mask.shape)
    for i in xrange(rav_mask.shape[0]):
        if rav_mask[i] > .5:
            c = l_coefs.pop(0)
            z_mask[i] += c
    z_mask = z_mask.reshape(resampled_mask.shape)
    aff = nib.load('/home/nick/Desktop/dki_stuff/modalities/dmean/dmeanCON001.nii').get_affine()
    finished_mask = nib.Nifti1Image(z_mask,affine=aff)
    return finished_mask

# def p_maxes(path):
#     df = pd.read_pickle(path)
#     df = df.drop_duplicates()
#     df = df.drop_duplicates(('gamma','alpha','fit_intercept'))
#     print '{} Parameters Fit'.format(len(df))
#     print '{}% Maximum Accuracy'.format(np.round(df.sort('results_mean',ascending=False).iloc[0].results_mean,4) * 100)
#     return df, df.sort('results_mean',ascending=False)[df.results_mean == max(df.results_mean)].sort('n_features')
