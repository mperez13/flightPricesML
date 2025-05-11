# Databricks notebook source
# MAGIC %md
# MAGIC # Gradient Boosted Trees (GBT)
# MAGIC
# MAGIC The following uses two regression models to predict flight fares based on various features.

# COMMAND ----------

# Importing required classes from PySpark's ML library
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.sql.functions import col, isnull, lit, udf
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.sql.functions import * 
from pyspark.ml.evaluation import RegressionEvaluator 
import time
import pandas as pd
from pyspark.ml.feature import MinMaxScaler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

# Set for running PySpark in CLI (Command Line Interface) mode
PYSPARK_CLI = True # False means we are not running in CLI mode
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# Load CSV dataset into a Spark DataFrame and display it
# File location and type
file_location = "/user/mpere110/flights_LAX.csv"
file_type = "csv"

# CSV options for reading the data
infer_schema = "true" # Option to infer the schema (data types) of columns automatically
first_row_is_header = "true" # Option indicating that the first row contains column names
delimiter = "," # Delimiter used in the CSV file

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Display the DataFrame contents
df.show()

# COMMAND ----------

# print the schema of the DataFrame df, showing the data types of each column.
df.printSchema()

# COMMAND ----------

# Function to convert ISO 8601 (e.g. PT10H30M) duration string to total number of minutes. This will be used to extract data from specified columns

def parse_duration(duration):
    import re
    match = re.match(r'P(?:(\d+)D)?(?:T(?:([\d]+H)?([\d]+M)?)?)?', duration)

    if not match:
        raise ValueError(f"Invalid duration format: {duration}")
    
    days = int(match.group(1)) if match.group(1) else 0

    hours = int(match.group(2)[:-1]) if match.group(2) and match.group(2) != '' else 0
    minutes = int(match.group(3)[:-1]) if match.group(3) and match.group(3) != '' else 0

    # Convert the duration to minutes
    total_minutes = days * 24 * 60 + hours * 60 + minutes

    return total_minutes

# Register UDF
parse_duration_udf = udf(parse_duration, IntegerType())

# COMMAND ----------

# Create new columns with extracted data and handle null values

# flightDayOfYear: extract the day of the year
df = df.withColumn("flightDayOfYear", dayofyear(col("flightDate").cast("date")))

# travelDurationMin: apply UDF to get travel duration in minutes
df = df.withColumn("travelDurationMin", parse_duration_udf(col("travelDuration")).cast(DoubleType()))

# flightMonth: extract month from flightDate
df = df.withColumn("flightMonth", month("flightDate"))

# flightYear: extract flight year
df = df.withColumn("flightYear", year("flightDate"))

# SearchDayoftheYear: extract Day from searchDate
df = df.withColumn("SearchDayoftheYear", dayofyear(col("searchDate").cast("date")))

# Handles any potential null values by filling the values with 0
df = df.na.fill(0)  # Fill nulls if any

# COMMAND ----------

# Drop the columns that are not going to be used
df = df.drop("legID","segmentsDepartureTimeEpochSeconds","segmentsArrivalTimeEpochSeconds","segmentsArrivalAirportCode","segmentsDepartureAirportCode","segmentsAirlineName","segmentsAirlineCode","segmentsEquipmentDescription","segmentsDurationInSeconds","segmentsDistance","segmentsCabinCode","segmentsCabinCode","segmentsDistance","segmentsDurationInSeconds","segmentsArrivalTimeRaw","segmentsDepartureTimeRaw","segmentsAirlineCode:","startingAirport",)

# COMMAND ----------

# print the schema of the DataFrame, df, showing the data types of each column
df.printSchema()
#  display the first 10 rows of the DataFrame in a tabular format
#df.show(10)

# COMMAND ----------

# Create a StringIndexer to convert the categorical feature "destinationAirport" into numerical indices. Print the schema of the newly created dataFrame
indexer = StringIndexer(inputCols=["destinationAirport"], outputCols=["destinationAirportId"])
df = indexer.fit(df).transform(df)
df.printSchema()

# COMMAND ----------

# names of the features (columns) from the DataFrame that will be used as features for training a machine learning model
cols = [
    "flightDayOfYear",
    "elapsedDays",
    "isBasicEconomy",
    "isRefundable",
    "isNonStop",
    "destinationAirportId", 
    "seatsRemaining",
    "totalTravelDistance",
    "travelDurationMin",
    "baseFare"   
]

# COMMAND ----------

# Split the data in DataFrame, df2,  into training and testing sets using random sampling

splits = df.randomSplit([0.7,0.3]) 
train = splits[0] # assigns the first element of the splits array to the variable. set the training set
test = splits[1] # assigns the second element of the splits array to the variable. set the testing set
train_rows = train.count()
test_rows = test.count() 
print("Training Rows:", train_rows, "Testing Rows:", test_rows) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

# Feature Important for Gradient Boosted
assemblerGBT = VectorAssembler(inputCols = cols, outputCol = "features")

minMaxGBT = MinMaxScaler(inputCol = assemblerGBT.getOutputCol(), outputCol = "normFeatures")

gbt_FI = GBTRegressor(labelCol = "totalFare", featuresCol = "normFeatures")

pipeline_FI_GBT = Pipeline(stages=[assemblerGBT, minMaxGBT, gbt_FI])

modelGBT = pipeline_FI_GBT.fit(train)

model_FI_GBT = modelGBT.stages[-1]

featureImpGBT = pd.DataFrame(list(zip(assemblerGBT.getInputCols(),
                                   model_FI_GBT.featureImportances)),
                          columns=["normFeatures", "importance"])
featureImpGBT.sort_values(by="importance", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model : Gradient Boosted Decision Trees

# COMMAND ----------

# Create a VectorAssembler to combine input columns into a single feature vector
assemblerGBT =  VectorAssembler(
    inputCols=cols,
    outputCol="features"
)

# COMMAND ----------

# Create a Gradient Boosted Trees Regressor (GBT) model
gbt = GBTRegressor(labelCol="totalFare", featuresCol="features")

# COMMAND ----------

# Create a parameter grid for tuning the Gradient Boosted Trees (GBT) model
paramGridGBT = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [7])  # Test tree depths 5 and 10
             .addGrid(gbt.maxBins, [25])  # Number of bins
             .addGrid(gbt.maxIter, [15])  # Number of trees
             .build())

# COMMAND ----------

# Create a RegressionEvaluator to evaluate the model's performance using R2 score
gbt_evaluator = RegressionEvaluator(predictionCol="prediction", 
                                    labelCol="totalFare",
                                    metricName="r2")

# COMMAND ----------


# Create a pipeline for the Gradient Boosted Trees (GBT) model, combining feature assembly and model fitting
gbt_pipeline = Pipeline(stages=[assemblerGBT, gbt])


# COMMAND ----------

# Start time to check how long it takes for TrainValidationSplit for Gradient Boosted Trees to run
start = time.time()

# COMMAND ----------

# Create a TrainValidationSplit for hyperparameter tuning and model evaluation using the GBT pipeline
gbt_tv = TrainValidationSplit(estimator=gbt_pipeline, 
                          evaluator= gbt_evaluator,
                          estimatorParamMaps=paramGridGBT,
                          trainRatio=0.75)

# COMMAND ----------

# Train the Gradient Boosted Trees (GBT) model using TrainValidationSplit and the training data
model_gbt_tv = gbt_tv.fit(train)

# COMMAND ----------

end = time.time()
phrase = 'Gradient Boosted Trees using TrainValidationSplit'
print('{} takes {} seconds'.format(phrase, end-start))

# COMMAND ----------

# Use the trained model to make predictions on the test set
prediction_gbt_tv = model_gbt_tv.transform(test)

# Select the relevant columns from the prediction output: features, predicted values, and actual target (totalFare)
predicted_gbt_tv = prediction_gbt_tv.select("features", "prediction", "totalFare")

# Display the predictions (features, predicted totalFare, and actual totalFare)
predicted_gbt_tv.show() 

# COMMAND ----------

# Create a RegressionEvaluator to calculate R-squared (coefficient of determination) for GBT
gbt_evaluator_tv_r2 = RegressionEvaluator(labelCol="totalFare",
                                             predictionCol="prediction",
                                             metricName="r2") 
r2_gbt_tv = gbt_evaluator_tv_r2.evaluate(prediction_gbt_tv)
 

# Create RegressionEvaluator to calculate RMSE (Root Mean Square Error) for GBT
gbt_evaluator_tv_rmse = RegressionEvaluator(labelCol="totalFare",
                                               predictionCol="prediction",
                                               metricName="rmse") 
rmse_gbt_tv = gbt_evaluator_tv_rmse.evaluate(prediction_gbt_tv);

# Print the results of R2 and RMSE for Gradient Boosted Decision Trees model using TrainValidationSplit
print("Train Validation Split for Gradient Boosted Trees")
print("Coefficient of Determination (R2) for GBT: ", r2_gbt_tv)
print("Root Mean Square Error (RMSE) for GBT: ", rmse_gbt_tv)

# COMMAND ----------

# Start time to check how long it takes for CrossValidation for Gradient Boosted Trees to run
start = time.time()

# COMMAND ----------

# Define the CrossValidator for GBT
gbt_cv = CrossValidator(estimator=gbt_pipeline, 
                        evaluator= gbt_evaluator, 
                        estimatorParamMaps=paramGridGBT, 
                        numFolds=3) 

# COMMAND ----------

# Train the Gradient Boosted Trees (GBT) model using Cross Validation and the training data
model_gbt_cv = gbt_cv.fit(train)

# COMMAND ----------

end = time.time()
phrase = 'Gradient Boosted Trees using CrossValidation'
print('{} takes {} seconds'.format(phrase, end-start))

# COMMAND ----------

# Use the trained model to make predictions on the test set
prediction_gbt_cv = model_gbt_cv.transform(test) 

# Select the relevant columns from the prediction output: features, predicted values, and actual target (totalFare)
predicted_gbt_cv = prediction_gbt_cv.select("features", "prediction", "totalFare")

# Display the predictions (features, predicted totalFare, and actual totalFare)
predicted_gbt_cv.show()


# COMMAND ----------

# Create a RegressionEvaluator to calculate R-squared (coefficient of determination) for GBT
gbt_evaluator_cv_r2 = RegressionEvaluator(labelCol="totalFare",
                                          predictionCol="prediction",metricName="r2") 
r2_gbt_cv = gbt_evaluator_cv_r2.evaluate(prediction_gbt_cv)
 
# Create RegressionEvaluator to calculate RMSE (Root Mean Square Error) for GBT
gbt_evaluator_cv_rmse = RegressionEvaluator(labelCol="totalFare",
                                           predictionCol="prediction",metricName="rmse") 
rmse_gbt_cv = gbt_evaluator_cv_rmse.evaluate(prediction_gbt_cv);

# Print the results of R2 and RMSE for Gradient Boosted Decision Trees model using Cross Validation
print("Cross Validation for Gradient Boosted Decision Trees")
print("Coefficient of Determination (R2) for Gradient Boosted Decision Trees: ", r2_gbt_cv)
print("Root Mean Square Error (RMSE) for Gradient Boosted Decision Trees: ", rmse_gbt_cv)