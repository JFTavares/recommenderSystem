from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from datetime import datetime

# Use movielens-100K
data = Dataset.load_builtin('ml-100k')

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
'reg_all': [0.4, 0.6]}

start = datetime.now()
#dispara o algoritmo
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
print('Tempo decorrido:{}\n'.format(datetime.now()-start))

# best RMSE score
print('Score RMSE:')
print(gs.best_score['rmse'])
print('Score MAE :')
print(gs.best_score['mae'])

# combination of parameters that gave the best RMSE score
print('Melhores parametros RMSE')
print(gs.best_params['rmse'])
print('Melhores parametros MAE')
print(gs.best_params['mae'])
