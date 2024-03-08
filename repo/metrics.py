import numpy as np
import pandas as pd

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
      
      dist = cdist(nx.reshape(1, -1)[:, continuous_features], ncf_list[:, continuous_features], metric=metric)[0]

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

from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
 
def diversity(counterfactuals, metric = 'l2'):
    counterfactuals = pd.DataFrame(scale(counterfactuals), columns = counterfactuals.columns)
    dist = pairwise_distances(counterfactuals, metric = metric)
    return np.triu(dist,k = 1).sum() / counterfactuals.shape[0] / counterfactuals.shape[0]

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
    
    nX = scaler.transform(X)
    ncf_list = scaler.transform(cf_list)
    lof = LocalOutlierFactor(n_neighbors = 10, metric = 'l2').fit(nX)
    neigh_dist, neigh_ind = lof.kneighbors(ncf_list)
    return neigh_dist.mean(axis = 1)
    
    # nbr_plausibility = []
    # nX = scaler.transform(X)
    # ncf_list = scaler.transform(cf_list)
    # clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    # lof_values = []
    # for var in variable_features:
    #     clf.fit(np.vstack(nX[:, var]))
    #     lof_values.append(clf.predict(np.vstack(ncf_list[:, var])))
    # nbr_plausibility = np.array(lof_values)
    # nbr_plausibility = [(nbr_plausibility[:,i] < 0).sum() for i in range(len(cf_list))]
    # return nbr_plausibility
    
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
            diversity_ = diversity(cf_list, metric = 'l2')
            validity_mean = validity_.mean()
            validity_std = validity_.std()
            proximity_mean = proximity_.mean()
            proximity_std = proximity_.std()
            sparsity_mean = sparsity_.mean()
            sparsity_std = sparsity_.std()
            plausibility_domain_mean = np.mean(plausibility_domain_)
            plausibility_domain_std = np.std(plausibility_domain_)
            plausibility_lof_mean = np.mean(plausibility_lof_)
            plausibility_lof_std =  np.mean(plausibility_lof_)
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






import dice_ml
from dice_ml.utils import helpers

def CF_evaluation_DICE(df, model, y_val, f_indexes, 
                  mutable_attr, cat_cols, cont_cols, n, TARGET, res_df, seed, gen_method):
    for i in f_indexes:
        n_row = i
        max_nbr_cf = None
        variable_features = [df.columns.get_loc(c) for c in mutable_attr if c in df]
        continuous_features = [df.columns.get_loc(c) for c in cont_cols if c in df]
        categorical_features = [df.columns.get_loc(c) for c in cat_cols if c in df]

        dice_data = dice_ml.Data(dataframe=df,
                      continuous_features=cont_cols,
                      outcome_name=TARGET)
        dice_model = dice_ml.Model(model=model, backend="sklearn")
        explainer = dice_ml.Dice(dice_data,dice_model, method=gen_method)
        example = df.iloc[n_row:n_row+1:].drop(TARGET, axis=1)
        if gen_method == 'random':
            e1 = explainer.generate_counterfactuals(example, total_CFs=n, desired_class='opposite',
                                                                features_to_vary= mutable_attr,
                                                                proximity_weight=0.5,
                                                                diversity_weight=1.0, random_seed=seed)
        else:
            e1 = explainer.generate_counterfactuals(example, total_CFs=n, desired_class='opposite',
                                                    features_to_vary= mutable_attr,
                                                    proximity_weight=0.5,
                                                    diversity_weight=1.0)

        cf_df = e1.cf_examples_list[0].final_cfs_df.drop(TARGET, axis=1)

        for col in mutable_attr:
            cf_df[col] = cf_df[col].astype('int64')
        cf_list = np.array(cf_df)
        res_DICE = evaluate_cf_list(cf_list, example.values[0], model, y_val, variable_features, continuous_features,
                            categorical_features, df.drop(TARGET, axis=1))

        res_DICE['method'] = 'DICE'
        res_DICE['f_index'] = n_row

    #     cf_df['PREDICT'] = model.predict(cf_df)
        cf_df[TARGET] = model.predict(cf_df)
        example_df = df.iloc[n_row:n_row+1:].rename(index={n_row: f'F_{n_row}'})
    #     example_df['PREDICT'] = model.predict(example_df.drop([TARGET], axis = 1))
        res_df = pd.concat([res_df, pd.concat([pd.concat([example_df, cf_df.reset_index().drop('index', axis=1)]),
            pd.DataFrame(res_DICE)], axis=1).fillna('metrics')], axis=0)
        
    return res_df, cf_df

def standartize(factual, counterfactuals, df):
    scaler = StandardScaler(with_mean=True, with_std=True).fit(df)
    f = pd.DataFrame(scaler.transform(factual.values.reshape(1,-1)), columns = counterfactuals.columns)
    c = pd.DataFrame(scaler.transform(counterfactuals), columns = counterfactuals.columns)
    return f,c

def CF_evaluation_synth(df, synthetic_data, synthetic_method, model, y_val, f_indexes, 
                  mutable_attr,immutable_attr, cat_cols, cont_cols, k, TARGET, res_df):
    
    variable_features = [df.columns.get_loc(c) for c in mutable_attr if c in df]
    continuous_features = [df.columns.get_loc(c) for c in cont_cols if c in df]
    categorical_features = [df.columns.get_loc(c) for c in cat_cols if c in df]

    factuals = df.iloc[f_indexes]
    for i in range(factuals.shape[0]):
        factual = factuals.iloc[i,:]
        counterfactuals = synthetic_data.copy()
        counterfactuals = counterfactuals.drop_duplicates(keep='first')
        for j in immutable_attr:
            counterfactuals[i] = factual[j]
            
        counterfactuals[TARGET] = model.predict(counterfactuals)                # predicting target labels
        counterfactuals = counterfactuals[counterfactuals[TARGET] != y_val]        # selecting counterfactuals 
        # computing distances using l2 norm 
        if not counterfactuals.empty:
            f, c  = standartize(factual, counterfactuals, df)    
            counterfactuals['dist'] = np.linalg.norm((c - f.values.reshape(1, -1)).drop([TARGET], axis = 1), ord = 2, axis = 1) 
            # sorting and selecting top k counterfactuals
            counterfactuals = counterfactuals.sort_values(by = 'dist', ascending = True)
            counterfactuals = counterfactuals[0:k]
            counterfactuals.drop(['dist'], axis = 1, inplace = True)
            print(f'A total of {counterfactuals.shape[0] :3d} counterfactuals were found')
            cf_list = np.array(counterfactuals.drop(TARGET, axis=1))
            res = evaluate_cf_list(cf_list, factual.drop(TARGET, axis=0).values, model, y_val, variable_features, continuous_features,
                            categorical_features, df.drop(TARGET, axis=1))
    
            res['method'] = synthetic_method
            res['f_index'] = f_indexes[i]
            n_row = f_indexes[i]
            example_df = pd.DataFrame(factual).T.rename(index={n_row: f'F_{n_row}'})
            res_df = pd.concat([res_df, pd.concat([pd.concat([example_df, counterfactuals.reset_index().drop('index', axis=1)]),
                pd.DataFrame(res)], axis=1).fillna('metrics')], axis=0)

    return res_df, counterfactuals

def CF_evaluation_GCS(df, factual, synthetic_data, synthetic_method, model, y_val, f_indexes, 
                  mutable_attr, immutable_attr, cat_cols, cont_cols, k, TARGET, res_df):
                      
    variable_features = [df.columns.get_loc(c) for c in mutable_attr if c in df]
    continuous_features = [df.columns.get_loc(c) for c in cont_cols if c in df]
    categorical_features = [df.columns.get_loc(c) for c in cat_cols if c in df]
    counterfactuals = synthetic_data.copy()
    counterfactuals = counterfactuals.drop_duplicates(keep='first')
    for i in immutable_attr:
        counterfactuals[i] = factual[i]
        
    counterfactuals[TARGET] = model.predict(counterfactuals)                # predicting target labels
    counterfactuals = counterfactuals[counterfactuals[TARGET] != y_val]        # selecting counterfactuals 
    # computing distances using l2 norm 
    if not counterfactuals.empty:
        f, c  = standartize(factual, counterfactuals, df)    
        counterfactuals['dist'] = np.linalg.norm((c - f.values.reshape(1, -1)).drop([TARGET], axis = 1), ord = 2, axis = 1) 
        # sorting and selecting top k counterfactuals
        counterfactuals = counterfactuals.sort_values(by = 'dist', ascending = True)
        counterfactuals = counterfactuals[0:k]
        counterfactuals.drop(['dist'], axis = 1, inplace = True)
        print(f'A total of {counterfactuals.shape[0] :3d} counterfactuals were found')
        cf_list = np.array(counterfactuals.drop(TARGET, axis=1))
        res = evaluate_cf_list(cf_list, factual.drop(TARGET, axis=0).values, model, y_val, variable_features, continuous_features,
                        categorical_features, df.drop(TARGET, axis=1))
    
        res['method'] = synthetic_method
        res['f_index'] = f_indexes    
        n_row = f_indexes
        example_df = pd.DataFrame(factual).T.rename(index={n_row: f'F_{n_row}'})
        res_df = pd.concat([res_df, pd.concat([pd.concat([example_df, counterfactuals.reset_index().drop('index', axis=1)]),
            pd.DataFrame(res)], axis=1).fillna('metrics')], axis=0)

    return res_df, counterfactuals
