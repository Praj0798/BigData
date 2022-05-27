import getpass
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
from pyspark.sql.functions import * 
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import RankingMetrics
import pyspark as ps

def main(spark, netId):
    file_path = f"hdfs:/user/{netId}/"
    file_name = ["train_small.parquet", "test_small.parquet", "val_small.parquet"]
    schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG'
    train_file_path = file_path + file_name[0]
    val_file_path = file_path + file_name[2]
    df_train = spark.read.parquet(train_file_path, schema = schema)
    df_val = spark.read.option("header", "true").parquet(val_file_path, schema = schema)
    df_train = df_train.dropna()
    df_train.printSchema()
    df_train.createOrReplaceTempView('df_train')
    count_train = df_train.count()
    val_pop = df_val.groupBy(df_val.userId).agg(func.collect_list("movieId")) 
    train_pop = df_train.groupBy(df_train.movieId).agg({'rating':'avg'}).limit(count_train)
    count_df = df_train.groupBy(df_train.movieId).count().filter("`count` >= 100")
    merge_df= train_pop.join(count_df, train_pop.movieId == count_df.movieId, "inner").drop(count_df.movieId)
    merge_df = merge_df.orderBy('avg(rating)', ascending=False)
    pred_movies = train_pop.select("movieId").limit(100).toPandas()['movieId'].tolist()
    gd_truth = val_pop.select("collect_list(movieId)").toPandas()['collect_list(movieId)']
    val_label_score = [(pred_movies, row) for row in gd_truth]
    sc=spark.sparkContext
    eval_df = sc.parallelize(val_label_score)
    metrics = RankingMetrics(eval_df)
    MAP=metrics.meanAveragePrecision
    MAP_1=metrics.meanAveragePrecisionAt(100)
    Ndcg_10=metrics.ndcgAt(100)
    MAP_recall_1= metrics.recallAt(1)  
    MAP_recall_5= metrics.recallAt(5)
    MAP_recall_10= metrics.recallAt(100)
    print("The evaluation metrics are : ")
    print(f"MAP : {MAP}")
    print(f"MAP@100 : {MAP_1}")
    print(f"NDCG@10 : {Ndcg_10}")
    print(f"MAPRecall@1 : {MAP_recall_1}")
    print(f"MAPRecall@5 : {MAP_recall_5}")
    print(f"MAPRecall@100 : {MAP_recall_10}")

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('rec_sys').getOrCreate()
    # Call our main routine
    netID = getpass.getuser()

    main(spark, netID)

    
    

