#!/usr/bin/env python
# coding: utf-8 
import pyspark as pys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
from pyspark.sql.functions import * 
from pyspark.sql.functions import col
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName('split_data').getOrCreate()

# file_path = "/scratch/work/courses/DSGA1004-2021/movielens/"
file_path = f"hdfs:/user/psn5377/"
# folder_name = "/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/"
op_file_path = "./data/"
sep = "/"
schema_ratings = 'userId INT, movieId INT, rating FLOAT, timestamp LONG'
schema_movies = 'movieId INT, title STRING, genres STRING'
ratings = spark.read.option("header", "true").csv(file_path+"ratings.csv", schema= schema_ratings)
movies = spark.read.option("header", "true").csv(file_path+"movies.csv", schema= schema_movies)
# tags = spark.read.csv(file_path+"tags.csv")
# links = spark.read.csv(file_path+"links.csv")
ratings = ratings.dropna()
movies = movies.dropna()
print(ratings.show())
# ratings = ratings.join(movies, "movieId", "inner").select("userId", "movieId", "rating", "genres")
fractions_init = ratings.select("userId").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap()
seed = 1231
train_init = ratings.stat.sampleBy("userId", fractions_init, seed)
test = ratings.subtract(train_init)
fractions_fin = train_init.select("userId").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap()
train = train_init.stat.sampleBy("userId", fractions_fin, seed)
val = train_init.subtract(train)


train_init.coalesce(1).write.csv(file_path+"train_small.csv")
test.coalesce(1).write.csv(file_path+"test_small.csv")
val.coalesce(1).write.csv(file_path+"validate_real_small.csv")


