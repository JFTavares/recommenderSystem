
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-1m')
trainset, testset = train_test_split(data, test_size=.15)
algoritmo = SVD(n_epochs=20)
algoritmo.fit(trainset)

uid = str(103)  
iid = str(1499)  

pred = algoritmo.predict(uid, iid, r_ui=1, verbose=True)
test_pred = algoritmo.test(testset)

print("Avaliação RMSE: ")
accuracy.rmse(test_pred, verbose=True)

print("Avaliação MAE: ")
accuracy.mae(test_pred, verbose=True)
