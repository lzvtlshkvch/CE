import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def nbr_valid_cf(cf_list, b, y_val, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered
    val = idx*1
    return val


def perc_valid_cf(cf_list, b, y_val, k=None, y_desidered=None):
    n_val = nbr_valid_cf(cf_list, b, y_val, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res        

def nbr_changes_per_cf(x, cf_list, variable_features):
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j]:
                nbr_changes[i] += 1 if j in variable_features else 0.5
    return nbr_changes

def distance_l2(x, cf_list, continuous_features, metric='euclidean', scaler=None, X=None, agg=None):
    if scaler is not None:
      nx = scaler.transform([x])[0]
      ncf_list = scaler.transform(cf_list)
      
      dist = cdist(nx.reshape(1, -1)[:, continuous_features], ncf_list[:, continuous_features], metric=metric)

      if agg is None:
          return dist
          
      if agg == 'mean':
          return np.mean(dist)

      if agg == 'max':
          return np.max(dist)

      if agg == 'min':
          return np.min(dist)

      if agg == 'std':
          return np.std(dist)
    else:
      return np.nan
        

def diversity_l2(cf_list, continuous_features, metric='euclidean', scaler=None, X=None, agg=None):
    if scaler is not None:
        ncf_list = scaler.transform(cf_list)
        dist = pdist(ncf_list[:, continuous_features], metric=metric)
    
        if agg is None:
            return np.linalg.det(squareform(1/(dist+1)))
            
        if agg == 'mean':
            return np.mean(dist)
            
        if agg == 'max':
            return np.max(dist)
    
        if agg == 'min':
            return np.min(dist)
            
        if agg == 'std':
            return np.std(dist)
    else:
        return np.nan

def plausibility_domain(cf_list, X, variable_features):
    nbr_plausibility = []
    for var in variable_features:
        min_var = X.iloc[:, var].min()
        max_var = X.iloc[:, var].max()
        nbr_plausibility.append(((cf_list[:, var] >= min_var) & (cf_list[:, var] <= max_var))*1)
    nbr_plausibility = np.array(nbr_plausibility)
    nbr_plausibility = [nbr_plausibility[:,i].mean() for i in range(len(cf_list))]
    return nbr_plausibility

def plausibility_lof(x, cf_list, X, variable_features, scaler):
    # nX = scaler.transform(X)
    # ncf_list = scaler.transform(cf_list)
    # clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    # clf.fit(nX)
    # lof_values = clf.predict(ncf_list)
    # # lof_values_nof = clf.negative_outlier_factor_
    
    nbr_plausibility = []
    nX = scaler.transform(X)
    ncf_list = scaler.transform(cf_list)
    clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    lof_values = []
    for var in variable_features:
        clf.fit(np.vstack(nX[:, var]))
        lof_values.append(clf.predict(np.vstack(ncf_list[:, var])))
    nbr_plausibility = np.array(lof_values)
    nbr_plausibility = [(nbr_plausibility[:,i] < 0).count() for i in range(len(cf_list))]
    return nbr_plausibility
    
def evaluate_cf_list(cf_list, x, model, y_val, variable_features, continuous_features,
                     categorical_features, X):
                         
    scaler = StandardScaler()
    scaler.fit(X)
    # scaler.transform([x])[0]
                         
    nbr_cf_ = len(cf_list)

    if nbr_cf_ > 0:    
        y_pred = model.predict(X)
        
        validity_ = nbr_valid_cf(cf_list, model, y_val, y_desidered=None)
        proximity_ = distance_l2(x, cf_list, continuous_features, metric='euclidean', scaler=scaler, X=None)
        sparsity_ = nbr_changes_per_cf(x, cf_list, variable_features)
        plausibility_domain_ = plausibility_domain(cf_list, X, variable_features)
        plausibility_lof_ = plausibility_lof(x, cf_list, X, variable_features, scaler)
        
        if len(cf_list) > 1:
            diversity_ = diversity_l2(cf_list, continuous_features, metric='euclidean', scaler=scaler, X=None, agg=None)
            validity_mean = validity_.mean()
            validity_std = validity_.std()
            proximity_mean = proximity_.mean()
            proximity_std = proximity_.std()
            sparsity_mean = sparsity_.mean()
            sparsity_std = sparsity_.std()
            plausibility_domain_mean = plausibility_domain_.mean()
            plausibility_domain_std =  plausibility_domain_.std()
            plausibility_lof_mean = plausibility_lof_.mean()
            plausibility_lof_std =  plausibility_lof_.std()
        else:
            diversity_ = 0.0
            validity_mean = 0.0
            validity_std = 0.0
            proximity_mean = 0.0
            proximity_std = 0.0
            sparsity_mean = 0.0
            sparsity_std = 0.0
            plausibility_domain_mean = 0.0
            plausibility_domain_std =  0.0
            plausibility_lof_mean = 0.0
            plausibility_lof_std =  0.0
    
        res = {
            'validity': validity_,
            'proximity': proximity_,
            'sparsity': sparsity_,
            'plausibility_domain': plausibility_domain_,
            'plausibility_lof': plausibility_lof_,
            'diversity': diversity_,
            'validity_mean': validity_mean,
            'validity_std': validity_std,
            'proximity_mean': proximity_mean,
            'proximity_std': proximity_std,
            'sparsity_mean': sparsity_mean,
            'sparsity_std': sparsity_std,
            'plausibility_domain_mean': plausibility_domain_mean,
            'plausibility_domain_std': plausibility_domain_std,
            'plausibility_lof_mean': plausibility_lof_mean,
            'plausibility_lof_std': plausibility_lof_std,
        }

    else:
        res = {
            'validity': np.nan,
            'proximity': np.nan,
            'sparsity': np.nan,
            'plausibility_domain': np.nan,
            'plausibility_lof': np.nan,
            'diversity': np.nan,
            'validity_mean': np.nan,
            'validity_std': np.nan,
            'proximity_mean': np.nan,
            'proximity_std': np.nan,
            'sparsity_mean': np.nan,
            'sparsity_std': np.nan,
            'plausibility_domain_mean': np.nan,
            'plausibility_domain_std': np.nan,
            'plausibility_lof_mean': np.nan,
            'plausibility_lof_std': np.nan,
            
        }

    return res

columns = [ 'validity',
            'proximity',
            'sparsity',
            'plausibility_domain',
            'plausibility_lof',
            'diversity',
            'validity_mean',
            'validity_std',
            'proximity_mean',
            'proximity_std',
            'sparsity_mean',
            'sparsity_std',
            'plausibility_domain_mean',
            'plausibility_domain_std',
            'plausibility_lof_mean',
            'plausibility_lof_std',
]
