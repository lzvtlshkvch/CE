import numpy as np
import pandas as pd

def plot_categorical(var, ax):
  good = df[df.credit_appr == 1][var].value_counts()
  bad  = df[df.credit_appr == 0][var].value_counts()
  pd.DataFrame({'good' : good.sort_index(), 'bad' : bad.sort_index()}).plot.bar(stacked = False, ax = ax);
  ax.set_title(var + f': {df[["credit_appr", var]].corr().iloc[0,1] :.3f}');

def plot_continuous(var, ax):
  sns.kdeplot(data = df, x = var, hue="credit_appr", hue_order = [1,0], ax = ax)
  ax.set_title(var + f': {df[["credit_appr", var]].corr().iloc[0,1] :.3f}');
  ax.set_xlabel('');
