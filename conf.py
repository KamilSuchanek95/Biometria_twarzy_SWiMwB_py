import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipdb

df = pd.read_csv("/home/kamil/Pulpit/Biometria_twarzy_SWiMwB_py/models/lbph/test_BioID_LBPH-2020-06-10_10-20-11.csv")

conf = pd.crosstab(df['true'],df['predict'])

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.figure()
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

plot_confusion_matrix(conf)

ipdb.set_trace()