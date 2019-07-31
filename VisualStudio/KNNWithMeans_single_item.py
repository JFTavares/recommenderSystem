
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
data = Dataset.load_builtin('ml-1m')
trainset, testset = train_test_split(data, test_size=.15)

# Configura o algoritmo. K = número de vizinhos. Name = Tipo de medida de similiradade. User based = filtragem por usuário ou item.

algoritmo = KNNWithMeans(k=20, sim_options={'name': 'cosine', 'user_based': False, 'verbose' : True})

algoritmo.fit(trainset)

# Selecionamos o usuário e o filme que será analisado
# User 49. Tem entre 18 e 24 anos. É programador e mora em Huston, Texas
#uid = str(49) 
# Filme visto e avaliado: Negotiator, The (1998)::Action|Thriller. Avaliação 4
#iid = str(2058)  # raw item id
#outro usuario 103::1499::1  --> nao gostou
uid = str(49)
iid = str(2058)
# get a prediction for specific users and items. MUDAR 1 ou 4 cfme o usuario/item
pred = algoritmo.predict(uid, iid, r_ui=1, verbose=True)

# run the trained model against the testset
test_pred = algoritmo.test(testset)

# Avalia RMSE
print("Avaliação RMSE: ")
accuracy.rmse(test_pred, verbose=True)

# Avalia MAE
print("Avaliação MAE: ")
accuracy.mae(test_pred, verbose=True)