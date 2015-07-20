from sklearn.linear_model import ElasticNet, Lasso, LassoLars, LassoLarsIC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
import pandas as pd
import pickle
from numpy import *
import numpy as np
import os
from sklearn.metrics import f1_score, make_scorer

data_frame_file_path = 'masked_matrices.pickle'  


#alpha_range = np.load('alphas_{}_fit.npy'.format(mode))
#gamma_range = np.load('gammas_{}_fit.npy'.format(mode))

alpha_range = np.logspace(-10,4,30)
alpha_range = np.array([alpha_range])

gamma_range = np.logspace(-15,5,50)


mode = ('kmean','fa','dmean')


masked_matrices = pd.read_pickle(data_frame_file_path)
print 'dataframe @ {} successfuly loaded'.format(data_frame_file_path)


# creating SUBJECTxFEATURE matrix

inputs = []
for m in mode:
    inputs.append( np.vstack((masked_matrices.ix[m].cons,masked_matrices.ix[m].pats)))

x_input = np.hstack(inputs)





masked_matrices = pd.read_pickle(data_frame_file_path)
print 'dataframe @ {} successfuly loaded'.format(data_frame_file_path)


# creating SUBJECTxFEATURE matrix
print 'training data for modality {} includes {} examples and {} features'.format(mode,*x_input.shape)
# creating y output vector (this is ugly fix this) (yyy must have shape of (trn_exmpl,-))
# creating y output vector (this is ugly fix this) (yyy must have shape of (trn_exmpl,-))
y_len = map(len,masked_matrices.ix[mode[0]])
n_samples = np.sum(y_len)
y_labels_vector = np.vstack((np.array([0] * y_len[0]).reshape((-1,1)),np.array([1] * y_len[1]).reshape((-1,1)))).reshape(n_samples)
print 'Y label vector is {} elements, with {} False and {} True\n'.format(len(y_labels_vector),y_len[0],y_len[1])
print 'y_labels_vector.shape : {}       ( Should be (n_samples,) )'.format(y_labels_vector.shape)
print 'x_input.shape         : {}  ( Should be (n_samples,n_features) )'.format(x_input.shape)



pickle_path = '{}_Elastic_NET'.format(mode)
if os.path.isfile(pickle_path):
    Lasso_DataFrame = pd.read_pickle(pickle_path)
else:
    Lasso_DataFrame = pd.DataFrame(columns=['results_mean','alpha','l1_ratio','C','gamma','Normalize','fit_intercept','n_features'])
  

Lasso_DataFrame.to_pickle(pickle_path)



n_cpus = -1
l1 = 1 # 1 for Lasso
C = 1
norm = False
fit_int = True
x_input.shape



results = []
for al in alpha_range:
    x_input_to_svc = np.copy(x_input)
    en = ElasticNet(alpha=al,l1_ratio = l1,fit_intercept=fit_int,normalize=norm,copy_X = True,max_iter=1e6)
    en.fit(x_input_to_svc,y_labels_vector.astype(np.float))
    w = en.coef_
    w[w > 0] = 1
    w[w <= 0 ] = 0
    print 'Weights Shape : {}'.format(w.shape)
    for i in xrange(x_input_to_svc.shape[1]):
        x_input_to_svc[:,i] *= w[i]

    
    ## multiply x+inputs by w ####
    print '{} positive coefficients @ alpha = {}'.format(np.count_nonzero(w),al)
    if np.count_nonzero(w) > 0:
        feature_selection_svc = Pipeline([('svc', SVC(C=C))])


        ## creating param_grid for search
        param_grid = dict(svc__gamma=gamma_range)
        ## cross-val function
        cv = StratifiedKFold(y_labels_vector,6)

        grid = GridSearchCV(feature_selection_svc, param_grid=param_grid,cv=cv,verbose=6,n_jobs=n_cpus,scoring = make_scorer(f1_score))
        grid.fit(x_input_to_svc,y_labels_vector)
        for g in grid.grid_scores_:
        	Lasso_DataFrame.loc[ len(Lasso_DataFrame) + 1 ] = [g[1],al,l1,C,g[0]['svc__gamma'],norm,fit_int,np.count_nonzero(w)]
        Lasso_DataFrame.to_pickle(pickle_path)
    
 


    
Lasso_DataFrame.to_pickle(pickle_path)