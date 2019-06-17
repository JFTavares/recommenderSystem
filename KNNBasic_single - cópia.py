from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
data = Dataset.load_builtin('ml-1m')
trainset, testset = train_test_split(data, test_size=.15)

# Configura o algoritmo. K = número de vizinhos. Name = Tipo de medida de similiradade. User based = filtragem por usuário ou item.

print("Usando o algoritmo KNNBasic com 50 vizinhos")
print("Algoritmo de similiraridade: Pearson")
algoritmo = KNNBasic(k=50, sim_options={'name': 'pearson', 'user_based': True, 'verbose' : True})

algoritmo.fit(trainset)

# Selecionamos o usuário e o filme que será analisado
# User 49. Tem entre 18 e 24 anos. É programador e mora em Huston, Texas
uid = str(49)  
# Filme visto e avaliado: Negotiator, The (1998)::Action|Thriller. Avaliação 4
iid = str(2058)  # raw item id

# get a prediction for specific users and items.
print("Predição de avaliação: ")
pred = algoritmo.predict(uid, iid, r_ui=4, verbose=True)

# run the trained model against the testset
test_pred = algoritmo.test(testset)

# Avalia RMSE
print("Avaliação RMSE: ")
accuracy.rmse(test_pred, verbose=True)

# Avalia MAE
print("Avaliação MAE: ")
accuracy.mae(test_pred, verbose=True)