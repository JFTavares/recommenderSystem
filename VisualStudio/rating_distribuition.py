import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as pet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBaseline, evaluate, accuracy
from surprise.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',\
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror',\
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1', parse_dates=True)

ratings['unix_timestamp'] = ratings['unix_timestamp'].apply(datetime.fromtimestamp)
ratings.columns = ['user_id', 'movie_id', 'rating', 'time']
ratings.head(10)

ratings['rating'].hist(bins=9)