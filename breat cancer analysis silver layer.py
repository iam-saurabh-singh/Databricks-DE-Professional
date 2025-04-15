# Databricks notebook source
import pandas as pd

url = "https://raw.githubusercontent.com/a-forty-two/stackroute-13March-Data_Engineering-Batch5/refs/heads/main/data.csv"

df = pd.read_csv(url)

spark_df = spark.createDataFrame(df)


# COMMAND ----------

display(spark_df)

# COMMAND ----------

import pandas as pd
from datetime import datetime
import os
 
# Config
source_file = "https://raw.githubusercontent.com/a-forty-two/stackroute-13March-Data_Engineering-Batch5/refs/heads/main/data.csv"
bronze_dir = "bronze"
bronze_file = os.path.join(bronze_dir, "breast_cancer_data_raw.csv")
 
# Create bronze dir if it doesn't exist
os.makedirs(bronze_dir, exist_ok=True)
 
# Load and store raw data
df = pd.read_csv(source_file)
df['ingestion_timestamp'] = datetime.now()
df.to_csv(bronze_file, index=False)

# COMMAND ----------

df_bronze = pd.read_csv("bronze/breast_cancer_data_raw.csv")
display(df_bronze)

# COMMAND ----------

mod_df_bronze = df_bronze.drop(columns=['Unnamed: 32'])
display(mod_df_bronze)

# COMMAND ----------

df_silver = mod_df_bronze.copy()
df_silver.columns = [col.strip().upper().replace(' ', '_') for col in df_silver.columns]
display(df_silver)

# COMMAND ----------

df_silver.drop_duplicates(inplace=True)
display(df_silver)

# COMMAND ----------

df_silver = df_silver.drop_duplicates(subset=['ID'])
display(df_silver)

# COMMAND ----------

num_cols = df_silver.select_dtypes(include=['float64', 'int64']).columns
df_silver[num_cols] = df_silver[num_cols].fillna(0)
display(num_cols)
display(df_silver)

# COMMAND ----------

def round_decimals(df, decimal_places=2):
    float_cols = df_silver.select_dtypes(include=['float64', 'float32']).columns
    df_silver[float_cols] = df_silver[float_cols].round(decimal_places)
    return df_silver

# COMMAND ----------

df_silver = round_decimals(df_silver)
display(df_silver)

# COMMAND ----------

diagnosis_m = df_silver[df_silver['DIAGNOSIS'] == 'M']
diagnosis_b = df_silver[df_silver['DIAGNOSIS'] == 'B']
display(diagnosis_m)
display(diagnosis_b)

# COMMAND ----------

silver_dir = "silver"
os.makedirs(silver_dir, exist_ok=True)
diagnosis_m.to_csv(os.path.join(silver_dir, "breast_cancer_data_cleaned_m.csv"), index=False)
diagnosis_b.to_csv(os.path.join(silver_dir, "breast_cancer_data_cleaned_b.csv"), index=False)

# COMMAND ----------

df_gold_m = pd.read_csv("silver/breast_cancer_data_cleaned_m.csv")
df_gold_b = pd.read_csv("silver/breast_cancer_data_cleaned_b.csv")


# COMMAND ----------

df_gold_m = df_gold_m[['DIAGNOSIS','RADIUS_MEAN','TEXTURE_MEAN','PERIMETER_MEAN','AREA_MEAN']]

# COMMAND ----------

display(df_gold_m)

# COMMAND ----------

df_gold_summary_m = df_gold_m.groupby('DIAGNOSIS').agg({
    'RADIUS_MEAN': 'mean',
    'TEXTURE_MEAN': 'mean',
    'PERIMETER_MEAN': 'mean',
    'AREA_MEAN': 'mean'
}).reset_index()

# COMMAND ----------

display(df_gold_summary_m)

# COMMAND ----------

df_gold_summary_b = df_gold_b.groupby('DIAGNOSIS').agg({
    'RADIUS_MEAN': 'mean',
    'TEXTURE_MEAN': 'mean',
    'PERIMETER_MEAN': 'mean',
    'AREA_MEAN': 'mean'
}).reset_index()

# COMMAND ----------

display(df_gold_summary_b)

# COMMAND ----------

union_gold = df_gold_summary_m.append(df_gold_summary_b)
display(union_gold)

# COMMAND ----------

gold_dir = "gold"
os.makedirs(gold_dir, exist_ok=True)
union_gold.to_csv(os.path.join(gold_dir, "breast_cancer_data_gold.csv"), index=False)
