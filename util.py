import getpass
from pyspark.sql.functions import *
from pyspark.sql import SparkSession


def main(spark, netId):
    file_path = f"hdfs:/user/{netId}/"
    file_name = ["train_large.csv", "test_large.csv", "validate_small.csv"]
    schema = 'userId INT, movieId INT, rating FLOAT, timestamp LONG'
    train_file_path = file_path + file_name[0]
    val_file_path = file_path + file_name[2]
    test_file_path = file_path + file_name[1]
    df_train = spark.read.csv(train_file_path, schema = schema)
    df_val = spark.read.option("header", "true").csv(val_file_path, schema = schema)
    df_test = spark.read.option("header", "true").csv(test_file_path, schema = schema)
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()
    sorted_train = df_train.sort("userId", ascending= True)
    sorted_val = df_val.sort("userId", ascending= True)
    sorted_test = df_test.sort("userId", ascending= True)
    print("Training set: ")
    print(sorted_train.show())
    print("Testing set: ")
    print(sorted_test.show())
    print("Validation set: ")
    print(sorted_val.show())
    sorted_train.write.parquet("train_large.parquet")
    sorted_val.write.parquet("val_large.parquet")
    sorted_test.write.parquet("test_small.parquet")



if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('rec_sys').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
