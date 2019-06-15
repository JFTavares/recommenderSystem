from surprise import Dataset, evaluate
from surprise import KNNBasic
from collections import defaultdict
import os
import io
import csv


def loadTrainPredict():

    data = Dataset.load_builtin("ml-100k")
    trainingSet = data.build_full_trainset()

    sim_options = {
        'name': 'cosine',
        'user_based': False
    }
    
    knn = KNNBasic(sim_options=sim_options)

    knn.fit(trainingSet)

    testSet = trainingSet.build_anti_testset()

    predictions = knn.test(testSet)

    return predictions


def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name


def salvaResultado(recomendToUser):
    if os.path.isfile('resultado.csv'):
        os.remove('resultado.csv') 

    with open('resultado.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['User','Recomend 1', 'Recomend 2', 'Recomend 3'])
        for row in recomendToUser:
            csv_out.writerow(row)

def main():

    tupla = []
    lista = ()
    predictions = loadTrainPredict()

    top3_recommendations = get_top3_recommendations(predictions)
    rid_to_name = read_item_names()

    for uid, user_ratings in top3_recommendations.items():
        tupla.append(rid_to_name[iid] for (iid, _) in user_ratings)
        lista = list(tupla)
        lista.append(uid)
    
    recomendToUser = lista
    salvaResultado(recomendToUser)

if __name__ == "__main__":
    main()