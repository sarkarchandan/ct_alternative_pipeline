from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType,StructField,IntegerType
from pyspark.sql.functions import from_json,col


if __name__ == "__main__":
    odometrySchema: StructType = StructType(fields= [
        StructField(name="id", dataType=IntegerType(), nullable=False),
        StructField(name="rot_int", dataType=IntegerType(), nullable=False),
        StructField(name="angle_start", dataType=IntegerType(), nullable=False),
        StructField(name="angle_end", dataType=IntegerType(), nullable=False),
        StructField(name="img_dim", dataType=IntegerType(), nullable=False),
        StructField(name="pad_len", dataType=IntegerType(), nullable=False),
    ])
    spark: SparkSession = SparkSession \
        .builder.appName("data_abstractor") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df: DataFrame = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "example_data") \
    .option("delimeter",",") \
    .option("startingOffsets", "earliest") \
    .load() 

    df.printSchema()

    df1 = df.selectExpr(
        "CAST(value AS STRING)").select(from_json(col("value"),
        odometrySchema).alias("data")).select("data.*")
    df1.printSchema()

    df1.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .start() \
    .awaitTermination()
