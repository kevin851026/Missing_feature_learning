import json
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
plt.figure(100)
plt.plot([1,2,3],[1,2,3])
plt.savefig('aa.jpg')
# plt.close('all')
plt.figure(figsize=(12, 6))
# plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
file = ['model_base3_2_share']
for i in file:
    with open(i + '.json','r',encoding='utf-8') as file:
        chart_data = json.load(file)
        plt.plot(chart_data['epoch'],chart_data['val_acc'],label="val_acc")
plt.grid(True,axis="y",ls='--')
plt.legend(loc= 'best')
plt.xlabel('epoch',fontsize=20)
plt.savefig('aaa.jpg')
plt.close('all')
print('hi')