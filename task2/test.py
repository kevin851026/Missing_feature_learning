import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file = pd.read_csv('train.csv', encoding='utf-8')
file2 = pd.read_csv('test.csv', encoding='utf-8')
plt.figure(0)
plt.rcParams['figure.figsize'] = (9.0, 6.0)
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
plt.savefig('b')
for i in file.columns.tolist()[1:-1]:
    plt.figure()
    plt.rcParams['figure.figsize'] = (9.0, 6.0)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    data = [round(i,1) for i in file.loc[:, [i]].values.reshape(-1)]
    print(max(data),' ',min(data))
    data = [i for i in data if i <5 and i >-5]
    plt.hist(data,bins=40 ,alpha=.4,width=0.2)
    data = [round(i,1) for i in file2.loc[:, [i]].values.reshape(-1)]
    print(max(data),' ',min(data))
    data = [i for i in data if i <5 and i >-5]
    plt.hist(data,bins=40, alpha=.4,width=0.2)
    plt.xticks(np.linspace(-5,5,21))
    plt.savefig(i+'b')
# data = [round(i,1) for i in file.loc[:, ['F1']].values.reshape(-1) if i <10 and i >-10]
# plt.hist(data, alpha=.4)

# plt.show()