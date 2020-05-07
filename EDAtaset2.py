import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sb

df = pd.read_csv("C:/Users/kdani/Documents/naranja-team/heart.csv")
df.head()
df.corr()
df.shape
df['age'].mean() #mean

fig, ages_h = plt.subplots()
ages_h.hist(df['age'])
ages_h.set_title('Histograma de edaes')
ages_h.set_xlabel('Edad')
ages_h.set_ylabel('Frec')
fig # histograma de edades
plt.show()

####### box plot ##########

sb.boxplot('cp', 'age', data = df)
plt.show()


df['chol'].describe()
sb.distplot(df['chol'], color = 'r', bins = 100, hist_kws = {'alpha' : 0.3})

list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64']) #seleccionamos todo los tipos de datos
df_num.head()

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8) #ploteo de todas las features
df.tail()

#buscamos correlaciones
df_num_corr = df_num.corr()['oldpeak'][:]
gfl = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("hay {} valores fuertmente correlacionados con oldpeak:\n{}".format(len(gfl), gfl))
#oldpeakST depression induced by exercise relative to rest
#slope the slope of the peak exercise ST segment
for i in range(0, len(df_num.columns), 5):
    sb.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['oldpeak'])

##heat map

corr = df_num.drop('slope', axis=1).corr()
plt.figure(figsize=(50, 50))

sb.heatmap(corr[(corr >= 0.5) | (corr <= -0.2)],
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

#######################
