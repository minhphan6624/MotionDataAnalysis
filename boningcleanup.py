import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

workerstats = pd.read_excel(r"C:\Users\roriu\OneDrive - Swinburne University\Uni\Semester 2\A.I\Group Project Stuff\AllWorkers.xlsx")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

workerstats = workerstats.drop_duplicates()  # Dropping duplicate data if any
unique_values = workerstats.nunique() # Gets quantity of unique values from dataset
constant_columns = unique_values[unique_values == 1] # Isolates all columns with only 1 value for the whole column

workerstats_cleaned = workerstats.drop(columns=constant_columns.index) # Drops the columns that only have 1 unique value since we don't need em
workerstats_cleaned = workerstats_cleaned.drop(columns='Minute')
workerstats_cleaned.to_excel('workerstats_cleanedtest.xlsx', index=False)

#skewness = workerstats_cleaned.skew()

# for column in workerstats_cleaned.select_dtypes(include=['float64', 'int']).columns:
#     workerstats_cleaned[column] = winsorize(workerstats_cleaned[column], limits=[0.01, 0.01])

#print(workerstats_cleaned.describe())

# batch_size = 10
# numeric_columns = workerstats_cleaned.select_dtypes(include=['float64', 'int']).columns

#//////////////////////////////////////////////
# Should loop through the columns in batches
# for i in range(0, len(numeric_columns), batch_size):
#     plt.figure(figsize=(15, 8))
#     batch_columns = numeric_columns[i:i + batch_size]

#     sns.boxplot(data=workerstats_cleaned[batch_columns])
#     plt.xticks(rotation = 90)
#     print(plt.show())
# print("knife sharpness categories: ", workerstats_cleaned['Knife_Sharpness_Category'].unique())
# print(" sub activity: ", workerstats_cleaned['Label'].unique())
# print("Boning or Slicing: ", workerstats_cleaned['Main_Activity'].unique())

# corr_matrix_targeted = workerstats_cleaned.corr()['Knife_Sharpness_Category'].sort_values(ascending=False)
# top_corr_features = corr_matrix_targeted.index[:15]

# plt.figure(figsize=(15, 12))
# sns.heatmap(workerstats_cleaned[top_corr_features].corr(), annot=True, cmap='coolwarm')
# plt.title("Top features correlated to knife sharpness heatmap")
# plt.subplots_adjust(bottom=0.3)
# print(plt.show())

# correlation_matrix = workerstats_cleaned.corr()
# moderate_corr_features = correlation_matrix[(correlation_matrix > 0.3) & (correlation_matrix < 0.7)]

# correlation_matrix = workerstats_cleaned.corr()
# correlation_matrix.values[np.arange(correlation_matrix.shape[0]), np.arange(correlation_matrix.shape[1])] = np.nan

#//////////////////////////////////////////////////
# Find the column with the highest correlation for each feature
# max_correlations = correlation_matrix.max()
# high_corr_pairs = correlation_matrix.idxmax()

# max_corr_df = pd.DataFrame({
#     'Feature': correlation_matrix.columns,
#     'Highest Correlated Feature': high_corr_pairs,
#     'Correlation': max_correlations
# })

# max_corr_df.to_excel('max_correlations_test.xlsx', index=False)
# print(moderate_corr_features)

#//////////////////////////////////////////////////
# workerstats_cleaned.to_excel('updated_with_composites.xlsx', index=False)

# X = workerstats_cleaned.drop(columns=['Knife_Sharpness_Category', 'Label', 'Main_Activity'])
# y_knife = workerstats_cleaned['Knife_Sharpness_Category']
# y_main = workerstats_cleaned['Main_Activity']
# y_sub = workerstats_cleaned['Label']

# X_train, X_test, y_train, y_test = train_test_split(X, y_knife, test_size=0.2, random_state=42)
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)

# feature_importances = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': rf.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# print(feature_importances.head(10))

#max_corr_df = pd.read_excel(r'C:\Users\roriu\OneDrive - Swinburne University\Uni\Semester 2\A.I\Group Project Stuff\max_correlations_test.xlsx')




