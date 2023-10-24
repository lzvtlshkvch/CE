import numpy as np

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor

class DummyScaler:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def nbr_valid_cf(cf_list, b, y_val, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered
    val = np.sum(idx)
    return val


def perc_valid_cf(cf_list, b, y_val, k=None, y_desidered=None):
    n_val = nbr_valid_cf(cf_list, b, y_val, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_actionable_cf(x, cf_list, variable_features):
    nbr_actionable = 0
    nbr_features = cf_list.shape[1]
    for i, cf in enumerate(cf_list):
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_actionable += 1

    return nbr_actionable


def perc_actionable_cf(x, cf_list, variable_features, k=None):
    n_val = nbr_actionable_cf(x, cf_list, variable_features)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered

    nbr_valid_actionable = 0
    nbr_features = cf_list.shape[1]
    for i, cf in enumerate(cf_list):
        if not np.array(idx)[i]:
            continue
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_valid_actionable += 1

    return nbr_valid_actionable


def perc_valid_actionable_cf(x, cf_list, b, y_val, variable_features, k=None, y_desidered=None):
    n_val = nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_violations_per_cf(x, cf_list, variable_features):
    nbr_features = cf_list.shape[1]
    nbr_violations = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                nbr_violations[i] += 1
    return nbr_violations


def avg_nbr_violations_per_cf(x, cf_list, variable_features):
    return np.mean(nbr_violations_per_cf(x, cf_list, variable_features))

def std_nbr_violations_per_cf(x, cf_list, variable_features):
    return np.std(nbr_violations_per_cf(x, cf_list, variable_features))

def avg_nbr_violations(x, cf_list, variable_features):
    val = np.sum(nbr_violations_per_cf(x, cf_list, variable_features))
    nbr_cf, nbr_features = cf_list.shape
    return val / (nbr_cf * nbr_features)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):

    if metric == 'mad':
        mad = median_abs_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

    if agg == 'std':
        return np.std(dist)


def categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None):

    dist = cdist(x.reshape(1, -1)[:, categorical_features], cf_list[:, categorical_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)
    
    if agg == 'std':
        return np.std(dist)


def distance_l2j(x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_mh(x, cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


# basso e' meglio cambi con x
def nbr_changes_per_cf(x, cf_list, continuous_features):
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j]:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))
    
def std_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.std(nbr_changes_per_cf(x, cf_list, continuous_features))


def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    val = np.sum(nbr_changes_per_cf(x, cf_list, continuous_features))
    nbr_cf, _ = cf_list.shape
    return val / (nbr_cf * nbr_features)


def continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_abs_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        dist = pdist(cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)
        
    if agg == 'std':
        return np.std(dist)


def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None):

    dist = pdist(cf_list[:, categorical_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)
        
    if agg == 'std':
        return np.std(dist)


def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def count_diversity(cf_list, features, nbr_features, continuous_features):
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    for i in range(nbr_cf):
        for j in range(i+1, nbr_cf):
            for k in features:
                if cf_list[i][k] != cf_list[j][k]:
                    nbr_changes += 1 if j in continuous_features else 0.5
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features)


# piu alto e' meglio conta variet' tra cf
def count_diversity_all(cf_list, nbr_features, continuous_features):
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, continuous_features)


def euclidean_jaccard(x, A, continuous_features, categorical_features, ratio_cont=None):
    nbr_features = A.shape[1]
    dist_cont = cdist(x.reshape(1, -1)[:, continuous_features], A[:, continuous_features], metric='euclidean')
    dist_cate = cdist(x.reshape(1, -1)[:, categorical_features], A[:, categorical_features], metric='jaccard')
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def accuracy_knn_sklearn(x, cf_list, b, X_test, continuous_features, categorical_features, scaler, test_size=5):
    clf = KNeighborsClassifier(n_neighbors=1)
    X_train = np.vstack([x.reshape(1, -1), cf_list])
    y_train = b.predict(X_train)
    clf.fit(X_train, y_train)

    X_test_knn = select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler, test_size)
    y_test = b.predict(X_test_knn)
    y_pred = clf.predict(X_test_knn)

    return accuracy_score(y_test, y_pred)


def accuracy_knn_dist(x, cf_list, b, X_test, continuous_features, categorical_features, scaler, test_size=5):

    X_train = np.vstack([x.reshape(1, -1), cf_list])
    y_train = b.predict(X_train)

    nX_train = scaler.transform(X_train)

    X_test_knn, nX_test_knn = select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler,
                                              test_size, get_normalized=True)
    y_test = b.predict(X_test_knn)

    y_pred = list()
    for nx_test in nX_test_knn:
        dist = euclidean_jaccard(nx_test, nX_train, continuous_features, categorical_features)
        idx = np.argmin(dist)
        y_pred.append(y_train[idx])

    return accuracy_score(y_test, y_pred)


def select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler, test_size=5, get_normalized=False):
    y_val = b.predict(x.reshape(1, -1))
    y_test = b.predict(X_test)
    nx = scaler.transform(x.reshape(1, -1))
    nX_test = scaler.transform(X_test)
    dist_f = euclidean_jaccard(nx, nX_test[y_test == y_val], continuous_features, categorical_features)
    dist_cf = euclidean_jaccard(nx, nX_test[y_test != y_val], continuous_features, categorical_features)
    index_f = np.argsort(dist_f)[0][:test_size].tolist()
    index_cf = np.argsort(dist_cf)[0][:test_size].tolist()
    index = np.array(index_f + index_cf)
    if get_normalized:
        return X_test[index], nX_test[index]
    return X_test[index]


def lof(x, cf_list, X, scaler):
    X_train = np.vstack([x.reshape(1, -1), X])

    nX_train = scaler.transform(X_train)
    ncf_list = scaler.transform(cf_list)

    clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    clf.fit(nX_train)

    lof_values = clf.predict(ncf_list)
    return np.mean(np.abs(lof_values))


# differenza in predict proba con x -> piu alto e' megliio
def delta_proba(x, cf_list, b, agg=None):
    y_val = b.predict(x.reshape(1, -1))
    y_cf = b.predict(cf_list)
    deltas = np.abs(y_cf - y_val)

    if agg is None or agg == 'mean':
        return np.mean(deltas)

    if agg == 'max':
        return np.max(deltas)

    if agg == 'min':
        return np.min(deltas)

    if agg == 'std':
        return np.std(deltas)


def delta_proba_img_ts(x, cf_list, b, agg=None):
    y_val = b.predict(x.reshape((1,) + x.shape))
    y_cf = np.array([b.predict(cf.reshape((1,) + cf.shape)) for cf in cf_list])
    deltas = np.abs(y_cf - y_val)

    if agg is None or agg == 'mean':
        return np.mean(deltas)

    if agg == 'max':
        return np.max(deltas)

    if agg == 'min':
        return np.min(deltas)


def plausibility(x, bb, cf_list, X_test, y_pred, continuous_features_all,
                 categorical_features_all, X_train, ratio_cont):
    sum_dist = 0.0
    for cf in cf_list:
        y_cf_val = bb.predict(cf.reshape(1, -1))[0]
        X_test_y = X_test[y_cf_val == y_pred]
        # neigh_dist = exp.cdist(x.reshape(1, -1), X_test_y)
        neigh_dist = distance_mh(x.reshape(1, -1), X_test_y, continuous_features_all,
                        categorical_features_all, X_train, ratio_cont)
        idx_neigh = np.argsort(neigh_dist)[0]
        # closest_idx = closest_idx = idx_neigh[0]
        # closest = X_test_y[closest_idx]
        closest = X_test_y[idx_neigh]
        d = distance_mh(cf, closest.reshape(1, -1), continuous_features_all,
                        categorical_features_all, X_train, ratio_cont)
        sum_dist += d
    return sum_dist


def evaluate_cf_list(cf_list, x, bb, y_val, max_nbr_cf, variable_features, continuous_features_all,
                     categorical_features_all, X_train, X_test, ratio_cont, nbr_features):
    nbr_cf_ = len(cf_list)

    if nbr_cf_ > 0:
        # scaler = DummyScaler()
        # scaler.fit(X_train)
    
        y_pred = bb.predict(X_test)
    
        nbr_valid_cf_ = nbr_valid_cf(cf_list, bb, y_val)
        perc_valid_cf_ = perc_valid_cf(cf_list, bb, y_val, k=nbr_cf_)
        perc_valid_cf_all_ = perc_valid_cf(cf_list, bb, y_val, k=max_nbr_cf)
        nbr_actionable_cf_ = nbr_actionable_cf(x, cf_list, variable_features)
        perc_actionable_cf_ = perc_actionable_cf(x, cf_list, variable_features, k=nbr_cf_)
        perc_actionable_cf_all_ = perc_actionable_cf(x, cf_list, variable_features, k=max_nbr_cf)
        nbr_valid_actionable_cf_ = nbr_valid_actionable_cf(x, cf_list, bb, y_val, variable_features)
        perc_valid_actionable_cf_ = perc_valid_actionable_cf(x, cf_list, bb, y_val, variable_features, k=nbr_cf_)
        perc_valid_actionable_cf_all_ = perc_valid_actionable_cf(x, cf_list, bb, y_val, variable_features, k=max_nbr_cf)
        avg_nbr_violations_per_cf_ = avg_nbr_violations_per_cf(x, cf_list, variable_features)
        std_nbr_violations_per_cf_ = std_nbr_violations_per_cf(x, cf_list, variable_features)
        avg_nbr_violations_ = avg_nbr_violations(x, cf_list, variable_features)
    
        sum_dist = 0.0
        plausibility_dict = []
        for cf in cf_list:
            y_cf_val = bb.predict(cf.reshape(1, -1))[0]
            X_test_y = X_test.values[y_cf_val == y_pred]
            # neigh_dist = exp.cdist(x.reshape(1, -1), X_test_y)
            neigh_dist = distance_mh(x.reshape(1, -1), X_test_y, continuous_features_all,
                            categorical_features_all, X_train.values, ratio_cont)
            idx_neigh = np.argsort(neigh_dist)[0]
            # closest_idx = closest_idx = idx_neigh[0]
            # closest = X_test_y[closest_idx]
            closest = X_test_y[idx_neigh]
            d = distance_mh(cf, closest.reshape(1, -1), continuous_features_all,
                            categorical_features_all, X_train.values, ratio_cont)
            sum_dist += d
            plausibility_dict.append(d)
        plausibility_std = np.std(plausibility_dict)
        plausibility_sum = sum_dist
        
        # plausibility_sum = metrics.plausibility(x, bb, cf_list, X_test, y_val, continuous_features_all,
        #                                         categorical_features_all, X_train, ratio_cont)
        # plausibility_max_nbr_cf_ = plausibility_sum / max_nbr_cf
        plausibility_nbr_cf_ = plausibility_sum / nbr_cf_
        plausibility_nbr_valid_cf_ = plausibility_sum / nbr_valid_cf_ if nbr_valid_cf_ > 0 else plausibility_sum
        plausibility_nbr_actionable_cf_ = plausibility_sum / nbr_actionable_cf_ if nbr_actionable_cf_ > 0 else plausibility_sum
        plausibility_nbr_valid_actionable_cf_ = plausibility_sum / nbr_valid_actionable_cf_ if nbr_valid_actionable_cf_ > 0 else plausibility_sum
    
        distance_l2_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None)
        distance_mad_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train.values)
        distance_j_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard')
        distance_h_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming')
        distance_l2j_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont)
        distance_mh_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont)
    
        distance_l2_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
        distance_mad_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='min')
        distance_j_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='min')
        distance_h_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='min')
        distance_l2j_min_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
        distance_mh_min_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='min')
    
        distance_l2_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
        distance_mad_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='max')
        distance_j_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='max')
        distance_h_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='max')
        distance_l2j_max_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
        distance_mh_max_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='max')
        
        distance_l2_std_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='std')
        distance_mad_std_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='std')
        distance_j_std_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='std')
        distance_h_std_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='std')
        distance_l2j_std_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='std')
        distance_mh_std_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='std')
    
        avg_nbr_changes_per_cf_ = avg_nbr_changes_per_cf(x, cf_list, continuous_features_all)
        std_nbr_changes_per_cf_ = std_nbr_changes_per_cf(x, cf_list, continuous_features_all)
        avg_nbr_changes_ = avg_nbr_changes(x, cf_list, nbr_features, continuous_features_all)
    
        delta_ = delta_proba(x, cf_list, bb, agg='mean')
        delta_min_ = delta_proba(x, cf_list, bb, agg='min')
        delta_max_ = delta_proba(x, cf_list, bb, agg='max')
        delta_std_ = delta_proba(x, cf_list, bb, agg='std')

        if len(cf_list) > 1:
            diversity_l2_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None)
            diversity_mad_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train.values)
            diversity_j_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard')
            diversity_h_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming')
            diversity_l2j_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont)
            diversity_mh_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont)
    
            diversity_l2_min_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
            diversity_mad_min_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='min')
            diversity_j_min_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='min')
            diversity_h_min_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='min')
            diversity_l2j_min_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
            diversity_mh_min_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='min')
    
            diversity_l2_max_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
            diversity_mad_max_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='max')
            diversity_j_max_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='max')
            diversity_h_max_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='max')
            diversity_l2j_max_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
            diversity_mh_max_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='max')
            
            diversity_l2_std_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='std')
            diversity_mad_std_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train.values, agg='std')
            diversity_j_std_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='std')
            diversity_h_std_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='std')
            diversity_l2j_std_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='std')
            diversity_mh_std_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train.values, ratio_cont, agg='std')
    
        else:
            diversity_l2_ = 0.0
            diversity_mad_ = 0.0
            diversity_j_ = 0.0
            diversity_h_ = 0.0
            diversity_l2j_ = 0.0
            diversity_mh_ = 0.0
    
            diversity_l2_min_ = 0.0
            diversity_mad_min_ = 0.0
            diversity_j_min_ = 0.0
            diversity_h_min_ = 0.0
            diversity_l2j_min_ = 0.0
            diversity_mh_min_ = 0.0
    
            diversity_l2_max_ = 0.0
            diversity_mad_max_ = 0.0
            diversity_j_max_ = 0.0
            diversity_h_max_ = 0.0
            diversity_l2j_max_ = 0.0
            diversity_mh_max_ = 0.0

            diversity_l2_std_ = 0.0
            diversity_mad_std_ = 0.0
            diversity_j_std_ = 0.0
            diversity_h_std_ = 0.0
            diversity_l2j_std_ = 0.0
            diversity_mh_std_ = 0.0

        count_diversity_cont_ = count_diversity(cf_list, continuous_features_all, nbr_features, continuous_features_all)
        count_diversity_cate_ = count_diversity(cf_list, categorical_features_all, nbr_features, continuous_features_all)
        count_diversity_all_ = count_diversity_all(cf_list, nbr_features, continuous_features_all)

        # accuracy_knn_sklearn_ = accuracy_knn_sklearn(x, cf_list, bb, X_test, continuous_features_all,
        #                                              categorical_features_all, scaler, test_size=5)
        # accuracy_knn_dist_ = accuracy_knn_dist(x, cf_list, bb, X_test, continuous_features_all,
        #                                        categorical_features_all, scaler, test_size=5)
        
        # lof_ = lof(x, cf_list, X_train, scaler)

        res = {
            'nbr_cf': nbr_cf_,
            'nbr_valid_cf': nbr_valid_cf_,
            'perc_valid_cf': perc_valid_cf_,
            'perc_valid_cf_all': perc_valid_cf_all_,
            'nbr_actionable_cf': nbr_actionable_cf_,
            'perc_actionable_cf': perc_actionable_cf_,
            'perc_actionable_cf_all': perc_actionable_cf_all_,
            'nbr_valid_actionable_cf': nbr_valid_actionable_cf_,
            'perc_valid_actionable_cf': perc_valid_actionable_cf_,
            'perc_valid_actionable_cf_all': perc_valid_actionable_cf_all_,
            'avg_nbr_violations_per_cf': avg_nbr_violations_per_cf_,
            'std_nbr_violations_per_cf': std_nbr_violations_per_cf_,
            'avg_nbr_violations': avg_nbr_violations_,
            'distance_l2': distance_l2_,
            'distance_mad': distance_mad_,
            'distance_j': distance_j_,
            'distance_h': distance_h_,
            'distance_l2j': distance_l2j_,
            'distance_mh': distance_mh_,
            'avg_nbr_changes_per_cf': avg_nbr_changes_per_cf_,
            'std_nbr_changes_per_cf': std_nbr_changes_per_cf_,
            'avg_nbr_changes': avg_nbr_changes_,
        
            'distance_l2_min': distance_l2_min_,
            'distance_mad_min': distance_mad_min_,
            'distance_j_min': distance_j_min_,
            'distance_h_min': distance_h_min_,
            'distance_l2j_min': distance_l2j_min_,
            'distance_mh_min': distance_mh_min_,
        
            'distance_l2_max': distance_l2_max_,
            'distance_mad_max': distance_mad_max_,
            'distance_j_max': distance_j_max_,
            'distance_h_max': distance_h_max_,
            'distance_l2j_max': distance_l2j_max_,
            'distance_mh_max': distance_mh_max_,

            'distance_l2_std': distance_l2_std_,
            'distance_mad_std': distance_mad_std_,
            'distance_j_std': distance_j_std_,
            'distance_h_std': distance_h_std_,
            'distance_l2j_std': distance_l2j_std_,
            'distance_mh_std': distance_mh_std_,
        
            'diversity_l2': diversity_l2_,
            'diversity_mad': diversity_mad_,
            'diversity_j': diversity_j_,
            'diversity_h': diversity_h_,
            'diversity_l2j': diversity_l2j_,
            'diversity_mh': diversity_mh_,
        
            'diversity_l2_min': diversity_l2_min_,
            'diversity_mad_min': diversity_mad_min_,
            'diversity_j_min': diversity_j_min_,
            'diversity_h_min': diversity_h_min_,
            'diversity_l2j_min': diversity_l2j_min_,
            'diversity_mh_min': diversity_mh_min_,
        
            'diversity_l2_max': diversity_l2_max_,
            'diversity_mad_max': diversity_mad_max_,
            'diversity_j_max': diversity_j_max_,
            'diversity_h_max': diversity_h_max_,
            'diversity_l2j_max': diversity_l2j_max_,
            'diversity_mh_max': diversity_mh_max_,

            'diversity_l2_std': diversity_l2_std_,
            'diversity_mad_std': diversity_mad_std_,
            'diversity_j_std': diversity_j_std_,
            'diversity_h_std': diversity_h_std_,
            'diversity_l2j_std': diversity_l2j_std_,
            'diversity_mh_std': diversity_mh_std_,
        
            'count_diversity_cont': count_diversity_cont_,
            'count_diversity_cate': count_diversity_cate_,
            'count_diversity_all': count_diversity_all_,
            # 'accuracy_knn_sklearn': accuracy_knn_sklearn_,
            # 'accuracy_knn_dist': accuracy_knn_dist_,
            # 'lof': lof_,
        
            'delta': delta_,
            'delta_min': delta_min_,
            'delta_max': delta_max_,
            'delta_std': delta_std_,
        
            'plausibility_sum': plausibility_sum,
            'plausibility_std': plausibility_std,
            # 'plausibility_max_nbr_cf': plausibility_max_nbr_cf_,
            'plausibility_nbr_cf': plausibility_nbr_cf_,
            'plausibility_nbr_valid_cf': plausibility_nbr_valid_cf_,
            'plausibility_nbr_actionable_cf': plausibility_nbr_actionable_cf_,
            'plausibility_nbr_valid_actionable_cf': plausibility_nbr_valid_actionable_cf_,
        }

    else:
        res = {
            'nbr_cf': nbr_cf_,
            'nbr_valid_cf': 0.0,
            'perc_valid_cf': 0.0,
            'perc_valid_cf_all': 0.0,
            'nbr_actionable_cf': 0.0,
            'perc_actionable_cf': 0.0,
            'perc_actionable_cf_all': 0.0,
            'nbr_valid_actionable_cf': 0.0,
            'perc_valid_actionable_cf': 0.0,
            'perc_valid_actionable_cf_all': 0.0,
            'avg_nbr_violations_per_cf': np.nan,
            'std_nbr_violations_per_cf': np.nan,
            'avg_nbr_violations': np.nan,
            'distance_l2': np.nan,
            'distance_mad': np.nan,
            'distance_j': np.nan,
            'distance_h': np.nan,
            'distance_l2j': np.nan,
            'distance_mh': np.nan,
            'distance_l2_min': np.nan,
            'distance_mad_min': np.nan,
            'distance_j_min': np.nan,
            'distance_h_min': np.nan,
            'distance_l2j_min': np.nan,
            'distance_mh_min': np.nan,
            'distance_l2_max': np.nan,
            'distance_mad_max': np.nan,
            'distance_j_max': np.nan,
            'distance_h_max': np.nan,
            'distance_l2j_max': np.nan,
            'distance_mh_max': np.nan,
            'distance_l2_std': np.nan,
            'distance_mad_std': np.nan,
            'distance_j_std': np.nan,
            'distance_h_std': np.nan,
            'distance_l2j_std': np.nan,
            'distance_mh_std': np.nan,
            'avg_nbr_changes_per_cf': np.nan,
            'std_nbr_changes_per_cf': np.nan,
            'avg_nbr_changes': np.nan,
            'diversity_l2': np.nan,
            'diversity_mad': np.nan,
            'diversity_j': np.nan,
            'diversity_h': np.nan,
            'diversity_l2j': np.nan,
            'diversity_mh': np.nan,
            'diversity_l2_min': np.nan,
            'diversity_mad_min': np.nan,
            'diversity_j_min': np.nan,
            'diversity_h_min': np.nan,
            'diversity_l2j_min': np.nan,
            'diversity_mh_min': np.nan,
            'diversity_l2_max': np.nan,
            'diversity_mad_max': np.nan,
            'diversity_j_max': np.nan,
            'diversity_h_max': np.nan,
            'diversity_l2j_max': np.nan,
            'diversity_mh_max': np.nan,
            'diversity_l2_std': np.nan,
            'diversity_mad_std': np.nan,
            'diversity_j_std': np.nan,
            'diversity_h_std': np.nan,
            'diversity_l2j_std': np.nan,
            'diversity_mh_std': np.nan,
            'count_diversity_cont': np.nan,
            'count_diversity_cate': np.nan,
            'count_diversity_all': np.nan,
            # 'accuracy_knn_sklearn': 0.0,
            # 'accuracy_knn_dist': 0.0,
            # 'lof': np.nan,
            'delta': 0.0,
            'delta_min': 0.0,
            'delta_max': 0.0,
            'delta_std': 0.0,
    
            'plausibility_sum': 0.0,
            'plausibility_std': 0.0,
            'plausibility_max_nbr_cf': 0.0,
            'plausibility_nbr_cf': 0.0,
            'plausibility_nbr_valid_cf': 0.0,
            'plausibility_nbr_actionable_cf': 0.0,
            'plausibility_nbr_valid_actionable_cf': 0.0,
        }

    return res

columns = ['dataset',  'black_box', 'method', 'idx', 'k', 'known_train', 'search_diversity', 'metric',
        'time_train', 'time_test', 'runtime', 'variable_features_flag',
        'nbr_cf', 'nbr_valid_cf', 'perc_valid_cf', 'perc_valid_cf_all', 'nbr_actionable_cf', 'perc_actionable_cf',
        'perc_actionable_cf_all', 'nbr_valid_actionable_cf', 'perc_valid_actionable_cf',
        'perc_valid_actionable_cf_all', 'avg_nbr_violations_per_cf', 'std_nbr_violations_per_cf', 'avg_nbr_violations',
        'distance_l2', 'distance_mad', 'distance_j', 'distance_h', 'distance_l2j', 'distance_mh',
        'distance_l2_min', 'distance_mad_min', 'distance_j_min', 'distance_h_min', 'distance_l2j_min',
        'distance_mh_min', 'distance_l2_max', 'distance_mad_max', 'distance_j_max', 'distance_h_max',
        'distance_l2j_max', 'distance_l2_std', 'distance_mad_std', 'distance_j_std', 'distance_h_std',
        'distance_l2j_std', 'distance_mh_std', 'avg_nbr_changes_per_cf','std_nbr_changes_per_cf', 'avg_nbr_changes', 'diversity_l2',
        'diversity_mad', 'diversity_j', 'diversity_h', 'diversity_l2j', 'diversity_mh', 'diversity_l2_min',
        'diversity_mad_min', 'diversity_j_min', 'diversity_h_min', 'diversity_l2j_min', 'diversity_mh_min',
        'diversity_l2_max', 'diversity_mad_max', 'diversity_j_max', 'diversity_h_max', 'diversity_l2j_max',
        'diversity_mh_max', 'diversity_l2_std', 'diversity_mad_std', 'diversity_j_std', 'diversity_h_std', 'diversity_l2j_std',
        'diversity_mh_std', 'count_diversity_cont', 'count_diversity_cate', 'count_diversity_all',
        # 'accuracy_knn_sklearn', 'accuracy_knn_dist', 'lof', 
        'delta', 'delta_min', 'delta_max', 'delta_std', 'instability_si',
        'plausibility_sum','plausibility_std', 'plausibility_max_nbr_cf', 'plausibility_nbr_cf', 'plausibility_nbr_valid_cf',
        'plausibility_nbr_actionable_cf', 'plausibility_nbr_valid_actionable_cf'
]
