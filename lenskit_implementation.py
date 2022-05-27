#!/usr/bin/env python

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pyspark.sql.functions as func 
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
spark = SparkSession.builder.config("spark.driver.memory", "20g").config("spark.driver.maxResultSize", "20g").appName('rec_sys').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark = SparkSession.builder.appName('rec_sys').getOrCreate()
from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS, ALSModel
import pandas as pd
import numpy as np
file_path = f"hdfs:/user/mb7979/" 
file_name = ["train_large.csv", "test_large.csv", "validate_large.csv"]

df_train = spark.read.csv(file_path+file_name[0], schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG')
df_val = spark.read.option("header", "true").csv(file_path+file_name[2], schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG')
df_test = spark.read.csv(file_path+file_name[1], schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG')

df_train = df_train.dropna()
df_train.printSchema()

df_train.createOrReplaceTempView('df_train')

sc=spark.sparkContext

from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

df_train1 = df_train.withColumn('index_column_name', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1).toPandas()

df_train2=df_train1.rename(columns={"userId": "user", "movieId": "movie" ,"index_column_name" : "index" })

from lenskit import batch, topn, util, topn
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
get_ipython().run_line_magic('matplotlib', 'inline')

algo_ii = knn.ItemItem(0.1)
algo_als = als.BiasedMF(50) 

from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

df_train1 = df_train.withColumn('index_column_name', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1).toPandas()

df_val1 = df_val.withColumn('index_column_name', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1).toPandas()

df_train2 = df_train1.rename(columns={"userId":"user","movieId":"item","rating":"rating","timestamp":"timestamp","index_column_name":"index"},errors="raise")
df_val2 = df_val1.rename(columns={"userId":"user","movieId":"item","rating":"rating","timestamp":"timestamp","index_column_name":"index"},errors="raise")


def eval(aname, algo, df_train2):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(df_train2)
    users = df_val2.user.unique()
    recs = batch.recommend(fittable, users, 500)
    recs['Algorithm'] = aname
    return recs

all_recs = []
val_data2 = []
for df_train2, df_val2 in xf.partition_users(df_train2[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    val_data2.append(df_val2)
    all_recs.append(eval('ItemItem', algo_ii, df_train2))
    all_recs.append(eval('ALS', algo_als, df_train2))

all_recs = pd.concat(all_recs, ignore_index=True)
val_data2 = pd.concat(val_data2, ignore_index=True)
rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
results = rla.compute(all_recs, val_data2)
results.head(10)
results.groupby('Algorithm').ndcg.mean()
results.groupby('Algorithm').ndcg.mean().plot.bar()



