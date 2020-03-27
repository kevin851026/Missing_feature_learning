import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
data = pd.read_csv('./task1_re_train.csv', encoding='utf-8')
data.drop(['Id'], axis=1, inplace=True)
for i in range(len(data)):
	data.loc[i,'Class'] =ord(data.loc[i,'Class'])
print(data)
corrMatrix = data.corr()
print(corrMatrix)
chinese =matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/kaiu.ttf')
plt.figure(figsize= (18.0, 18.0))
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
sn.set(font=chinese.get_name())
sn.heatmap(corrMatrix, annot=True)
plt.savefig('dd')
plt.show()