import getpass
import pyspark.sql.functions as func 
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
# import pandas as pd
import csv

def helper_func(val_pop):
    movieIdList = []
    for row in val_pop.rdd.toLocalIterator():
        movieIdRow = []
        for rec_row in row:
            for rec_tuple in rec_row:
                movieId = rec_tuple[0]
                movieIdRow.append(movieId)
        movieIdList.append(movieIdRow)
    return movieIdList

def main(spark, netId, sc):
    file_path = f"hdfs:/user/{netId}/"
    file_name = ["train_small.parquet", "test_small.parquet", "val_small.parquet"]
    schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG'
    train_file_path = file_path + file_name[0]
    val_file_path = file_path + file_name[2]
    test_file_path = file_path + file_name[1]
    df_train = spark.read.parquet(train_file_path, schema = schema)
    df_val = spark.read.option("header", "true").parquet(val_file_path, schema = schema)
    df_test = spark.read.option("header", "true").parquet(test_file_path, schema = schema)
    df_train = df_train.dropna()
    df_train.printSchema()
    df_train.createOrReplaceTempView('df_train')
    seed = 4321
    maxIter = 10
    als = ALS(maxIter = maxIter, implicitPrefs = False, userCol='userId', itemCol='movieId',ratingCol='rating', coldStartStrategy="drop", seed = seed)
    valid_movie_list = df_val.groupby('userId').agg(func.collect_list('movieId'))
    ranks= [250]
    regParams= [0.1]
    errors = [[0]*len(ranks)]*len(regParams)
    models = [[0]*len(ranks)]*len(regParams)
    err = 0
    best_MAP = 0
    best_rank = -1
    i = 0
    for regs in regParams:
        j = 0
        for rank in ranks:
            als.setParams(rank = rank, regParam = regs)
            model = als.fit(df_train)
            users = valid_movie_list.select(als.getUserCol())
            predicted_val=model.recommendForUserSubset(users,100)
            ground_truth = predicted_val.join(valid_movie_list, "userId", "inner").select('collect_list(movieId)')
            recs = predicted_val.join(valid_movie_list, "userId", "inner").select('recommendations')
            recs_list = helper_func(recs)
            gd_truth_list = []
            for row in ground_truth.rdd.toLocalIterator():
                row = list(row)
                gd_truth_list.append(row[0])
            valid_labels_and_scores = list(zip(recs_list, gd_truth_list))
            eval_df = sc.parallelize(valid_labels_and_scores)
            metrics = RankingMetrics(eval_df)
            MAP=metrics.meanAveragePrecision
            MAP_1=metrics.meanAveragePrecisionAt(100)
            errors[i][j] = MAP
            models[i][j] = model
            print('For maxIter %s, rank %s, regularization parameter %s the MAP is %s, the MAP@100 is %s' % (maxIter, rank, regs, MAP, MAP_1))

            param_row = [rank, regs, MAP, MAP_1]
            with open("params.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(param_row)

            if MAP > best_MAP:
                min_error = MAP
                best_params = [i,j]
            j += 1
        i += 1
    als.setRegParam(regParams[best_params[0]]) #setting best value for rank to model
    als.setRank(ranks[best_params[1]]) #setting best model for regParam to model
    print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
    print ('The best model was trained with rank %s' % ranks[best_params[1]])
    my_model = models[best_params[0]][best_params[1]]
    my_model.itemFactors.write.parquet("item_factors.parquet")
  #predicting on test data with best value for params
    final_predict= my_model.transform(df_test)
    print(final_predict.show())






            


    # print("Before ALS")
    # # rank=10, maxIter=5, seed=42, regParam=regParam,
    
    
    # print("After ALS")

    
    
    # print("Print valid: ", valid_movie_list.show())
    # print("Print predictions: ", predicted_val.show())

    # ground_truth = predicted_val.join(valid_movie_list, "userId", "inner").select('collect_list(movieId)')
    # recs = predicted_val.join(valid_movie_list, "userId", "inner").select('recommendations')
    # print("After join")

    # recs_list = helper_func(recs)

    # gd_truth_list = []
    # for row in ground_truth.rdd.toLocalIterator():
    #     row = list(row)
    #     gd_truth_list.append(row[0])

    # print("AFter gd_truth_list")
    # valid_labels_and_scores = list(zip(recs_list, gd_truth_list))
    # print("Before valid_labels_and_scores movie list")
    # eval_df = sc.parallelize(valid_labels_and_scores)

    # metrics = RankingMetrics(eval_df)
    # MAP=metrics.meanAveragePrecision
    # MAP_1=metrics.meanAveragePrecisionAt(100)
    # Ndcg_10=metrics.ndcgAt(100)
    # MAP_recall_1= metrics.recallAt(1)  
    # MAP_recall_5= metrics.recallAt(5)
    # MAP_recall_10= metrics.recallAt(100)

    # print("The evaluation metrics are : ")
    # print(f"MAP : {MAP}")
    # print(f"MAP@100 : {MAP_1}")
    # print(f"NDCG@100 : {Ndcg_10}")
    # print(f"MAPRecall@1 : {MAP_recall_1}")
    # print(f"MAPRecall@5 : {MAP_recall_5}")
    # print(f"MAPRecall@100 : {MAP_recall_10}")


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('rec_sys').getOrCreate()
    sc=spark.sparkContext
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID, sc)