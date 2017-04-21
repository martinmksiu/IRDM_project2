import numpy as np
from numpy import genfromtxt

def dcg_at_k(r, k, method=1):

    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0


def ndcg_at_k(r, k, method=1):
 
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0
    return dcg_at_k(r, k, method) / dcg_max

# open the saved data
print("opening saved datafiles")  # y, id, output
truerel = genfromtxt('first.csv', delimiter=',', skip_header=1)
idd = genfromtxt('second.csv', delimiter=',', skip_header=1)
model = genfromtxt('third.csv', delimiter=',', skip_header=1)

values = set(map(lambda x:x[1], idd))
lists = [[int(y[0]) for y in idd if y[1]==x] for x in values]
#numbers = [ int(x) for x in numbers ]actual = []

aa = list(zip(truerel[:,0],np.argmax(truerel[:,1:5], axis=1), model[:,2]))

def getKey(item):
     return item[2]
 
data = []
for i in range(len(lists)):
    a2 = [aa[idx] for idx in lists[i]]  
    a2.sort(key=getKey,reverse=True)
    data.append(a2)

ndgc = []
for q in range(len(lists)):
    true_y = [float(i[1]) for i in data[q]]
    #true_y = np.asarray(true_y)
    #y_score = [i[2] for i in data[q]]
    #y_score = np.asarray(y_score)
    ndgc.append(ndcg_at_k(true_y, 10))
    
ndcgat10 = np.mean(ndgc)
print(ndcgat10)
