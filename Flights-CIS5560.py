# Databricks notebook source
# MAGIC %md
# MAGIC # Random Forest Regressor (RFR) and Gradient Boosted Trees (GBT)
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

# COMMAND ----------

# Set for running PySpark in CLI (Command Line Interface) mode
PYSPARK_CLI = False # False means we are not running in CLI mode
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# Load CSV dataset into a Spark DataFrame and display it
# File location and type
file_location = "/FileStore/tables/startingAiport_LAX_df.csv"
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
display(df)

# COMMAND ----------

# print the schema of the DataFrame df, showing the data types of each column.
df.printSchema()

# COMMAND ----------

# Function to convert ISO 8601 (e.g. PT10H30M) duration string to total number of minutes. This will be used to extract data from specified columns

def parse_duration(duration):
    import re
    match = re.match(r'PT(\d+H)?(\d+M)?', duration)
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    return hours * 60 + minutes

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
df.display(10)

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

assembler = VectorAssembler(inputCols = cols, outputCol = "features")

minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol = "normFeatures")

rf_FI = RandomForestRegressor(labelCol = "totalFare", featuresCol = "normFeatures")

pipeline_FI = Pipeline(stages=[assembler, minMax, rf_FI])

model = pipeline_FI.fit(train)

model_FI = model.stages[-1]

featureImp = pd.DataFrame(list(zip(assembler.getInputCols(),
                                   model_FI.featureImportances)),
                          columns=["normFeatures", "importance"])
featureImp.sort_values(by="importance", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### \#1  Model : Random Forest Regressor

# COMMAND ----------

# Create a VectorAssembler to combine multiple column into a single feature vector
assemblerRF =  VectorAssembler(
    inputCols=cols,
    outputCol="features"
)

# COMMAND ----------

# Create RandomForestRegressor model
# 'totalFare' is the target variable that will be predicted
rf = RandomForestRegressor(labelCol='totalFare')

# COMMAND ----------

# Create the parameter grid for RandomForest Regressor
paramGridRF = (ParamGridBuilder()
             .addGrid(rf.numTrees, [10,20])  # Grid search over two values of numTrees: 10 and 20
             .addGrid(rf.maxDepth, [5])  # Grid search over one value of maxDepth: 5
             .build())

# COMMAND ----------


# # Create a RegressionEvaluator to assess model performance using the R2 metric
rf_evaluator = RegressionEvaluator(
    predictionCol="prediction", 
    labelCol="totalFare",
    metricName="r2")

# COMMAND ----------

# Defines a pipeline to convert the categorical feature "destinationAirport" into a numerical representation using StringIndexer
rf_pipeline = Pipeline(stages=[assemblerRF, rf])


# COMMAND ----------

# Start time to check how long it takes for cross validator for RandomForestRegressor to run
start = time.time()

# COMMAND ----------

# Create a CrossValidator to perform cross-validation for RandomForestRegressor
rf_cv = CrossValidator(estimator=rf_pipeline,  # The model pipeline that will be trained and validated
                       evaluator= rf_evaluator, # The evaluator used to assess model performance
                       estimatorParamMaps=paramGridRF, # The grid of hyperparameters to tune
                       numFolds=3) # The number of folds for cross-validation

# COMMAND ----------

# Fit the CrossValidator to the training data to find the best model based on cross-validation
model_rf_cv = rf_cv.fit(train)

# COMMAND ----------

end = time.time()
phrase = 'Random Forest Regression using Cross Validation'
print('{} takes {} seconds'.format(phrase, end-start))

# COMMAND ----------

# Use the best model found by cross-validation to make predictions on the test set
prediction_rf_cv = model_rf_cv.transform(test) 

# Select relevant columns ("features", "prediction", and "totalFare") from the predictions for analysis
predicted_rf_cv = prediction_rf_cv.select("features", "prediction", "totalFare") 

# Show the first few rows of the predicted results
predicted_rf_cv.show()

# COMMAND ----------

print("Cross Validation for Random Forest Regression")

# Create a RegressionEvaluator for evaluating the R2 (coefficient of determination)
rf_evaluator_cv_r2 = RegressionEvaluator(labelCol="totalFare", 
                                         predictionCol="prediction", 
                                         metricName="r2") 
r2_rf_cv = rf_evaluator_cv_r2.evaluate(prediction_rf_cv)
 

# Create a RegressionEvaluator for evaluating the RMSE (Root Mean Square Error)
rf_evaluator_cv_rmse = RegressionEvaluator(labelCol="totalFare", 
                                           predictionCol="prediction",
                                           metricName="rmse") 
rmse_rf_cv = rf_evaluator_cv_rmse.evaluate(prediction_rf_cv);

# Print the evaluated R2 and RMSE of the Random Forest Regression
print("Coefficient of Determination (R2) for Random Forest Regression: ", r2_rf_cv)
print("Root Mean Square Error (RMSE) for Random Forest Regression: ", rmse_rf_cv)


# COMMAND ----------

# Start time to check how long it takes for TrainValidationSplit for RandomForestRegressor to run
start = time.time()

# COMMAND ----------

# Create a TrainValidationSplit to perform model selection and tuning using training and validation sets
rf_tv = TrainValidationSplit(estimator=rf_pipeline, 
                          evaluator= rf_evaluator,
                          estimatorParamMaps=paramGridRF,
                          trainRatio=0.8) 

# COMMAND ----------

# Train the model using TrainValidationSplit on the training data
model_rf_tv = rf_tv.fit(train)

# COMMAND ----------

end = time.time()
phrase = 'Random Forest Regression using TrainValidationSplit'
print('{} takes {} seconds'.format(phrase, end-start))

# COMMAND ----------

# Use the trained model to make predictions on the test data
prediction_rf_tv = model_rf_tv.transform(test)

# Select relevant columns from the predictions
predicted_rf_tv = prediction_rf_tv.select("features", "prediction", "totalFare")

# COMMAND ----------

print("Train Validation Split for Random Forest Regression")

# Create an evaluator to compute the R2 score (Coefficient of Determination)
rf_evaluator_tv_r2 = RegressionEvaluator(labelCol="totalFare",
                                         predictionCol="prediction",
                                         metricName="r2") 
r2_rf_tv = rf_evaluator_tv_r2.evaluate(prediction_rf_tv)

# Create an evaluator to compute the Root Mean Square Error (RMSE)
rf_evaluator_tv_rmse = RegressionEvaluator(labelCol="totalFare", 
                                           predictionCol="prediction",
                                           metricName="rmse") 
rmse_rf_tv = rf_evaluator_tv_rmse.evaluate(prediction_rf_tv);

# Print the results of R2 and RMSE for the Random Forest model using TrainValidationSplit
print("Coefficient of Determination (R2) for Random Forest Regression: ", r2_rf_tv) 
print("Root Mean Square Error (RMSE) for Random Forest Regression: ", rmse_rf_tv)

# COMMAND ----------

# MAGIC %md
# MAGIC ### \#2  Model : Gradient Boosted Decision Trees

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

print("Train Validation Split for Gradient Boosted Trees")

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

print("Cross Validation for Gradient Boosted Decision Trees")

# Create a RegressionEvaluator to calculate R-squared (coefficient of determination) for GBT
gbt_evaluator_cv_r2 = RegressionEvaluator(labelCol="totalFare",
                                          predictionCol="prediction",metricName="r2") 
r2_gbt_cv = gbt_evaluator_cv_r2.evaluate(prediction_gbt_cv)
 
# Create RegressionEvaluator to calculate RMSE (Root Mean Square Error) for GBT
gbt_evaluator_cv_rmse = RegressionEvaluator(labelCol="totalFare",
                                           predictionCol="prediction",metricName="rmse") 
rmse_gbt_cv = gbt_evaluator_cv_rmse.evaluate(prediction_gbt_cv);

# Print the results of R2 and RMSE for Gradient Boosted Decision Trees model using Cross Validation
print("Coefficient of Determination (R2) for Gradient Boosted Decision Trees: ", r2_gbt_cv)
print("Root Mean Square Error (RMSE) for Gradient Boosted Decision Trees: ", rmse_gbt_cv)
