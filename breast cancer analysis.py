# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

url = "https://raw.githubusercontent.com/a-forty-two/stackroute-13March-Data_Engineering-Batch5/refs/heads/main/data.csv"

df = pd.read_csv(url)

spark_df = spark.createDataFrame(df)


# COMMAND ----------

spark_df.show()
print(df.info())
print(df.describe())

# COMMAND ----------

plt.style.use('seaborn-whitegrid')
sns.set_palette('Set2')

# COMMAND ----------

plt.figure(figsize=(4, 2))
sns.countplot(data=df, x='diagnosis')
plt.title('Diagnosis Count')
plt.xlabel('Diagnosis (M = Malignant, B = Benign)')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

plt.figure(figsize=(6, 3))
sns.histplot(data=df, x='radius_mean', hue='diagnosis', kde=True, bins=30, element='step')
plt.title('Radius Mean Distribution by Diagnosis')
plt.xlabel('Radius Mean')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

plt.figure(figsize=(6, 3))
sns.boxplot(data=df, x='diagnosis', y='area_mean')
plt.title('Area Mean by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Area Mean')
plt.show()



# COMMAND ----------

plt.figure(figsize=(12, 9))
corr = df.drop(columns=['id']).corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()



# COMMAND ----------

selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
sns.pairplot(df[selected_features], hue='diagnosis', corner=True)
plt.suptitle('Pairplot of Selected Features by Diagnosis', y=1.02)
plt.show()
