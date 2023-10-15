# import
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import xlrd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm, tqdm_notebook
import warnings
from scipy import stats
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import roc_auc_score
import re
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import itertools as it
import math
import time 
import scipy
from time import time
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from optbinning import OptimalBinning
from sklearn.metrics import f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

def plot_categorical(df, target, var, ax):
  good = df[df[target]== 1][var].value_counts()
  bad  = df[df[target]== 0][var].value_counts()
  pd.DataFrame({'good' : good.sort_index(), 'bad' : bad.sort_index()}).plot.bar(stacked = False, ax = ax);
  ax.set_title(var + f': {df[["credit_appr", var]].corr().iloc[0,1] :.3f}');

def plot_continuous(df, target, var, ax):
  sns.kdeplot(data = df, x = var, hue=target, hue_order = [1,0], ax = ax)
  ax.set_title(var + f': {df[[target, var]].corr().iloc[0,1] :.3f}');
  ax.set_xlabel('');

# Functions
def missing_values_table(df):
        mis_val = df.isnull().sum()
        zero_val = len(df) - df.astype(bool).sum(axis=0)
        unique_val = df.nunique()
        
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        zero_val_percent = 100 *(len(df) - df.astype(bool).sum(axis=0)) / len(df)
        unique_val_percent = 100 * df.nunique() / len(df)
        
        mis_val_table = pd.concat([mis_val, mis_val_percent, zero_val, zero_val_percent, unique_val, unique_val_percent], axis=1)
        
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missings', 1 : '% Missings', 2 : 'Zeroes', 3 : '% Zeroes', 4 : 'Uniques', 5 : '% Uniques'})
        
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values(
        '% Missings', ascending=False).round(1)
        
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        return mis_val_table_ren_columns
    
def Gini(df, x, target):
    model = sm.GLM(df[target], sm.add_constant(df[x], prepend = False) , family=sm.families.Binomial())
    result = model.fit()
    result_params = result.params[x]
    pred = result.predict()
    actual = df[target]
    fpr, tpr, thresholds = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    GINI = (2 * roc_auc) - 1
    pvalues = result.pvalues[0]
    return GINI, result_params, pvalues

# Basic combination selection functions
def calcComboParams(combo, current_gini, skipDynTest = False, returnAllParams = False, max_factor = 0):
  # функция проверки параметров комбинации и расчета Gini
  # по флагу returnAllParams = False возвращаем три значения: 
  # 1: отметка о прохождении стопов 
  # 2: отметка об улучшении Gini
  # 3: сам Gini
  # по флагу returnAllParams = True возвращаем два массива: 
  # 1: названия параметров (столбцов) 
  # 2: значения параметров
  
  # проверка на мультиколлиниарность
  if not isOkCorrelation(combo):
    return False, False, 0
  if not isOkVIF(combo):
    return False, False, 0
  
  # параметры регрессий на train и test
  model_train, x_train, coeffs_train, weights_train, pvals_train = buildRegression(combo, df_train, yTrain)
  model_test,  x_test,  coeffs_test,  weights_test,  pvals_test  = buildRegression(combo, df_test,  yTest)
  
  # общая проверка результатов регрессий
#   if not isOkCoeffs(coeffs_train) or not isOkCoeffs(coeffs_test):
#     return False, False, 0
  if not isOkWeights(weights_train) or not isOkWeights(weights_test):
    return False, False, 0
  if not isOkPvals(pvals_train) or not isOkPvals(pvals_test):
    return False, False, 0  
  
  # скоры train и test по коэффициентам train
  scores_train = model_train.predict(x_train)
  scores_test  = model_train.predict(x_test)
  
  # колмогоров-смирнов train и test
  ks_train = calcKolmogorov(scores_train, yTrain)
  ks_test  = calcKolmogorov(scores_test,  yTest)
  if not isOkKsLight(ks_train) or not isOkKsLight(ks_test):
    return False, False, 0
  
  # gini train и test
  gini_train = CalcGini(yTrain, scores_train)
  gini_test  = CalcGini(yTest,  scores_test)
  if not isOkGiniDiffer(gini_train, gini_test):
    return False, False, 0
  
  # итоговое сравнение Gini
  if returnAllParams:
    cols = ['Model']
    data = ['Model']
    max_factor = max(len(combo), max_factor)
    # факторы
    cols.append('const')
    cols.extend(['N' + str(i+1) for i in range(max_factor)])
    data.append('const')
    data.extend([i for i in combo])
    # --------train--------
    # коэффициенты train
    cols.append('train coeff const')
    cols.extend(['train coeff N' + str(i+1) for i in range(max_factor)])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    cols.append('train Good coeffs')
    data.extend([i for i in coeffs_train])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.append('OK' if isOkCoeffs(coeffs_train) else 'NO')
    # веса train
    cols.extend(['train weight N' + str(i+1) for i in range(max_factor)])
    cols.extend(['train MinWeight', 'train MaxWeight'])
    data.extend([i for i in weights_train])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.extend([min(weights_train), max(weights_train)])
    # pvals train
    cols.append('train pval const')
    cols.extend(['train pval N' + str(i+1) for i in range(max_factor)])
    cols.append('train Maxpval')
    data.extend([i for i in pvals_train])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.append(max(pvals_train))      
    # gini и ks train
    cols.extend(['train Gini','train KS'])
    data.extend([gini_train, ks_train])
    
    # --------test--------
    # коэффициенты test
    cols.append('test coeff const')
    cols.extend(['test coeff N' + str(i+1) for i in range(max_factor)])
    cols.append('test Good coeffs')
    data.extend([i for i in coeffs_test])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.append('OK' if isOkCoeffs(coeffs_test) else 'NO')
    # веса test
    cols.extend(['test weight N' + str(i+1) for i in range(max_factor)])
    cols.extend(['test MinWeight', 'test MaxWeight'])
    data.extend([i for i in weights_test])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.extend([min(weights_test), max(weights_test)])
    # pvals test
    cols.append('test pval const')
    cols.extend(['test pval N' + str(i+1) for i in range(max_factor)])
    cols.append('test Maxpval')
    data.extend([i for i in pvals_test])
    data.extend([math.nan for x in range(max_factor - len(combo))])
    data.append(max(pvals_test))      
    # gini и ks test
    cols.extend(['test KS'])
    data.extend([ks_test])
    # gini и ks test
    cols.extend(['test Gini','test KS'])
    data.extend([gini_test, ks_test])
    
    return cols, data
    
  else:
    
    if not skipDynTest:
      # new_gini = min(gini_train, gini_test)
      new_gini = gini_train
    else:
      # new_gini = min(gini_train, gini_test)
      new_gini = gini_train
      # new_gini = min(min(gini_dyn_train), min(gini_dyn_test))
    
    if new_gini < current_gini:
      return True, False, new_gini
    else:
      return True, True, new_gini

def getCorrelations(combo):
  # считаем корреляции
  # выводим два значения - максимальная корреляция по train и по test
  if len(combo) <= 1:
    return [1, 1]
  else:
    res = []
  
  # сортируем факторы, чтобы они были строго в порядке как в корреляциях
  idx, combo = sortCombo(combo)
  
  # отбираем таблицы с корреляциями
  b_trainClassCorr = trainClassCorr[np.ix_(idx, idx)]
  b_testClassCorr = testClassCorr[np.ix_(idx, idx)]
  
  # фильтруем и считаем
  for vals in (b_trainClassCorr, b_testClassCorr):
    np.absolute(vals)
    mask = np.ones(vals.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    res.append(vals[mask].max())
  
  return res

def getVIFs(combo):
  # считаем VIF
  # выводим два массива значений - VIF факторов по train и по test
  if len(combo) <= 1:
    return [[10], [10]]
  else:
    res = []
  
  # сортируем факторы, чтобы они были строго в порядке как в корреляциях
  idx, combo = sortCombo(combo)
  
  # отбираем таблицы с корреляциями
  b_trainRankCorr = trainRankCorr[np.ix_(idx, idx)]
  b_testRankCorr = testRankCorr[np.ix_(idx, idx)]
  
  # фильтруем и считаем
  for vals in (b_trainRankCorr, b_testRankCorr):
    np.absolute(vals)
    vals = np.linalg.inv(vals)
    mask = np.zeros(vals.shape, dtype=bool)
    np.fill_diagonal(mask, 1)
    res.append(vals[mask])
  
  return res

def sortCombo(combo):
  # сортируем факторы, чтобы они были строго в порядке как в корреляциях
  idx = [corr_cols.index(x) for x in combo]
  combo = [x for _,x in sorted(zip(idx,combo))]
  idx = [corr_cols.index(x) for x in combo]
  return idx, combo

def buildRegression(combo, data, yList):
  # построение регрессии и вывод основных параметров
  xList = data[combo]
  xList = sm.add_constant(xList)
  model = sm.Logit(yList, xList)
  results = model.fit(disp=False)
  
  # коэффициенты
  coeffs = results.params
  # веса
  weights = [x / sum(coeffs[1:]) for x in coeffs[1:]]
  # значимость
  pvals = results.pvalues
  
  return results, xList, coeffs, weights, pvals

def isOkCorrelation(combo, max_val = 0.7):
  return max(getCorrelations(combo)) < max_val

def isOkVIF(combo, max_val = 3):
  vifs = getVIFs(combo)
  a = max(vifs[0]) < max_val
  b = max(vifs[1]) < max_val
  return a and b

def isOkCoeffs(coeffs, max_val = 0):
  return max(coeffs[1:]) < max_val

def isOkWeights(weights, min_val = 0.05):
  return min(weights) > min_val

def isOkPvals(pvals, max_val = 0.05):
  return max(pvals) < max_val

def isOkGiniDiffer(gini_train, gini_test):
  rel_15 = abs(gini_test / gini_train - 1) < 0.15
  rel_20 = abs(gini_test / gini_train - 1) < 0.20
  abs_5  = abs(gini_test - gini_train)     < 0.05
  abs_10 = abs(gini_test - gini_train)     < 0.10
  
  return (rel_15 and abs_10) or abs_5

def calcKolmogorov(vals, defs):
  # считаем Колмогорова-Смирнова
  vals = vals*(-1)
  buckets = 100 # количество бакетов для теста
  
  if len(vals) == 0:
    return math.nan
  
  scores = np.array(vals)
  defs = np.array(defs)
  minScore = min(scores)
  maxScore = max(scores)
  dTotal = sum(defs)
  ndTotal = len(defs) - dTotal
  step = (maxScore-minScore) / (buckets-1)
  
  # строим карту бакетов
  boundList = np.array([minScore + x*step    for x in range(buckets)])
  # кумулятивные значения
  cumNDList = np.array([len(defs[scores<=boundList[x]])   for x in range(buckets)])
  cumDList  = np.array([sum(defs[scores<=boundList[x]])   for x in range(buckets)])
  cumNDList = cumNDList - cumDList
  # доли
  shNDList  = cumNDList / ndTotal
  shDList   = cumDList / dTotal
  # разница в долях
  difList   = shDList - shNDList
  # итоги теста
  maxDif    = max(difList)
  ksValue   = maxDif * \
              math.sqrt((ndTotal*dTotal) / (ndTotal+dTotal))
   
  
  return ksValue

def isOkKsLight(val):
  if val > 1.628:
    res = 'Green'
  elif val > 1.3581:
    res = 'Yellow'
  else:
    res = 'Red'
  return res  == 'Green'  

def GetScores(data,coeff):
  res = data.dot(coeff).values * (-1)
  return res

def CalcGini(actual_list, predicted_list):
  fpr, tpr, ttt = roc_curve(actual_list, predicted_list)
  return 2*auc(fpr, tpr) - 1

def step_selection(alg1, alg2, data, data_tt, col_list, target):
    max_col = []
    gini_list = []
    features = col_list.copy()
    max_ = 0
    for i in features: 
        alg1.fit(data[[i]], data[target].values) 
        Y_pred = pd.DataFrame(alg1.predict_proba(data[[i]]), columns = ['0_prob', '1_prob'])
        gini=2*roc_auc_score(data[target].values, Y_pred['1_prob'].values) - 1
        if gini > max_: 
            max_ = gini
            max_c = i
    max_col.append(max_c)
    features.remove(max_col[0])      
    gini_list.append(max_)
    k = 0
    for i in range(1, len(col_list)):
        max_ = 0
        for i in features: 
            col = max_col.copy()
            col.append(i)
            alg2.fit(data.loc[data.index.isin(list(data[col_list].index)),col], data.loc[data.index.isin(list(data[col_list].index)), target])
            Y_pred_oot = pd.DataFrame(alg2.predict_proba(data[col]), columns = ['0_prob', '1_prob'])
            fpr_oot, tpr_oot, _ = roc_curve(data[target].values, Y_pred_oot['1_prob'].values)
            gini = 2*roc_auc_score(data[target].values, Y_pred_oot['1_prob'].values) - 1
            if gini > max_: 
                max_ = gini
                max_c = i
        max_col.append(max_c)
        k += 1
        features.remove(max_col[k])      
        gini_list.append(max_)
        
    a = pd.DataFrame()
    a["features"] = max_col
    a["Gini"] = gini_list
    deviation = [0]
    for i in range(1,len(gini_list)):
        deviation.append(gini_list[i] - gini_list[i-1])
    a["Deviation"] = deviation
    plt.figure(figsize=(10,7))
    menMeans = []
    for i in range(0, len(a['features'])):
        menMeans.append(list(a['Gini'])[i] - list(a['Deviation'])[i])
    womenMeans = list(a['Deviation'])
    ind = np.arange(len(a['features']))
    width = 0.60
    p0 = plt.bar([0], list(a['Gini'])[0], width, color='darkgray')
    p2 = plt.bar(ind, womenMeans, width, color = 'darkgray',
                 bottom=menMeans)
    plt.ylabel('Gini')
    plt.title('Gini Uplift')
    plt.xticks(ind, a['features'])
    plt.xticks(rotation=90)
    plt.plot([list(a['Gini'])[0]] + list(a['Deviation'][1:len(list(a['Deviation']))]), marker = 'o',  color='black')
    plt.show()

    return a

def gini_feature(alg, data, col_name, target):
    alg.fit(data[[col_name]], data[target].values) 
    Y_pred = pd.DataFrame(alg.predict_proba(data[[col_name]]), columns = ['0_prob', '1_prob'])
    gini=2*roc_auc_score(data[target].values, Y_pred['1_prob'].values) - 1
    
    AUC = (gini + 1)*0.5
    Q_1 = AUC/(2 - AUC)
    Q_2 = 2*AUC**2/(1 + AUC)
    sorted_actual = data[target]

    st_dev_gini = 2*math.sqrt(abs((AUC*(1-AUC)+(sum(sorted_actual)-1)*(Q_1-AUC**2)+(len(sorted_actual)-sum(sorted_actual)-1)*(Q_2-AUC**2))/((len(sorted_actual)-sum(sorted_actual))*sum(sorted_actual))))
    
    return gini, st_dev_gini

def gini_graph(alg, data, target, col_list):
    gini = pd.DataFrame()
    gini["features"] = col_list
    a = []
    for i in col_list:
        a.append(gini_feature(alg, data, i, target)[0])
        
    gini["Gini Factor"] = a
    labels = col_list
    value = a
    position = np.arange(len(col_list))

    fig, ax = plt.subplots()

    ax.barh(position, value, color='#00bbe4')
    ax.set_yticks(position)
    ax.set_yticklabels(labels,
                       fontsize = 15)

    fig.set_figwidth(20)
    fig.set_figheight(12)

    plt.show()
    return gini

def roc_auc_curve(yTrain, yTest, preds_train, preds_test):
    fpr_train, tpr_train ,_ = roc_curve(yTrain, preds_train)
    fpr_test, tpr_test ,_ = roc_curve(yTest, preds_test)
    gini_train = 2*roc_auc_score(yTrain, preds_train) - 1
    gini_test = 2*roc_auc_score(yTest, preds_test) - 1
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr_train, tpr_train, label='train:Gini={!s:2.6s}'.format(gini_train*100))
    ax.plot(fpr_test, tpr_test, label='test:Gini={!s:2.6s}'.format(gini_test*100))
    ax.set(xlabel = 'False positive rate', ylabel = 'True positive rate', title = 'ROC curve')
    ax.legend(loc = 'lower right')
    print('ROC_curve')
    print('TRAIN: gini = ', gini_train)
    print('TEST: gini = ', gini_test)
    # plt.savefig(r'roc_auc_mort.png', format='png', dpi=100, bbox_inches = 'tight')
    plt.show()

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

# all_ginis = pd.DataFrame(columns = ['Model', 'Gini Train', 'Gini Test',
#                                     'Gini Variance, %', 'F1 train', 'F1 test',
#                                     'FPR Train', 'FNR Train', 'FPR Test', 'FNR Test'])

def metrics_clf(model, X_train, X_test, y_train, y_test, i, name, all_ginis):
  preds_train = model.predict_proba(X_train)
  preds_test = model.predict_proba(X_test)
  
  cm = confusion_matrix(y_train, [1 if i > 0.5 else 0 for i in preds_train[:,1]])
  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  
  FPR_train = FP/(FP+TN)
  FNR_train = FN/(TP+FN)
  
  cm = confusion_matrix(y_test, [1 if i > 0.5 else 0 for i in preds_test[:,1]])
  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  
  FPR_test = FP/(FP+TN)
  FNR_test = FN/(TP+FN)
  
  all_ginis.loc[i] = [name, CalcGini(y_train, preds_train[:,1]),\
                      CalcGini(y_test, preds_test[:,1]), 
                      (CalcGini(y_test, preds_test[:,1])-\
                       CalcGini(y_train, preds_train[:,1]))/CalcGini(y_train, preds_train[:,1])*100,
                      roc_curve(y_train, preds_train[:,1]),\
                      roc_curve(y_test, preds_test[:,1]),\
                      f1_score([1 if i > 0.5 else 0 for i in preds_train[:,1]], y_train),
                      f1_score([1 if i > 0.5 else 0 for i in preds_test[:,1]], y_test),
                     FPR_train, FNR_train, FPR_test, FNR_test]
  
