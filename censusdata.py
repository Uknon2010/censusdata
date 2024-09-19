# Question 1

import pandas as pd
import numpy as np
from pandas import DataFrame
df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")


print("Dataset Structure:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


missing_values = df.isnull().sum()
columns_with_missing = missing_values[missing_values > 0]
print("\nColumns with missing values:")
print(columns_with_missing)


income_stats = df['Income'].agg(['mean', 'median', 'std'])
totalpop_stats = df['TotalPop'].agg(['mean', 'median', 'std'])

print("\nStatistics for 'Income':")
print(income_stats)

print("\nStatistics for 'TotalPop':")
print(totalpop_stats)

#QUESTION 2
df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")


print("Dataset Structure (before cleaning):")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


df_cleaned = df.dropna()


print("\nDataset Structure (after cleaning):")
print(f"Number of rows: {df_cleaned.shape[0]}")
print(f"Number of columns: {df_cleaned.shape[1]}")


df_cleaned['MaleFemaleRatio'] = df_cleaned['Men'] / df_cleaned['Women']


median_income = df_cleaned['Income'].median()
print(f"\nMedian Income: ${median_income:.2f}")


df_filtered = df_cleaned[df_cleaned['Income'] > median_income]

#QUESTION 3

df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")

df_cleaned = df.dropna()

state_income_stats = df_cleaned.groupby('State')['Income'].agg(['mean', 'median', 'max', 'min'])

state_income_stats_sorted = state_income_stats.sort_values('mean', ascending=False)

print("Income Statistics by State (sorted by mean income):")
print(state_income_stats_sorted)

print("\nTop 5 States by Mean Income:")
print(state_income_stats_sorted.head())

print("\nBottom 5 States by Mean Income:")
print(state_income_stats_sorted.tail())

overall_stats = df_cleaned['Income'].agg(['mean', 'median', 'max', 'min'])
print("\nOverall Income Statistics:")
print(overall_stats)

state_income_stats_sorted['income_range'] = state_income_stats_sorted['max'] - state_income_stats_sorted['min']
state_income_stats_sorted['coefficient_of_variation'] = state_income_stats_sorted.apply(lambda row: row['mean'] / row.std(), axis=1)

print("\nTop 5 States with Highest Income Range:")
print(state_income_stats_sorted.nlargest(5, 'income_range')[['mean', 'median', 'max', 'min', 'income_range']])

print("\nTop 5 States with Lowest Income Range:")
print(state_income_stats_sorted.nsmallest(5, 'income_range')[['mean', 'median', 'max', 'min', 'income_range']])

print("\nTop 5 States with Highest Coefficient of Variation (Income Inequality):")
print(state_income_stats_sorted.nlargest(5, 'coefficient_of_variation')[['mean', 'median', 'coefficient_of_variation']])

print("\nTop 5 States with Lowest Coefficient of Variation (Income Equality):")
print(state_income_stats_sorted.nsmallest(5, 'coefficient_of_variation')[['mean', 'median', 'coefficient_of_variation']])

#QUESTION 4

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")

df_cleaned = df.dropna()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_cleaned, x='Income', bins=50, kde=True)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')

mean_income = df_cleaned['Income'].mean()
median_income = df_cleaned['Income'].median()
plt.axvline(mean_income, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_income:,.0f}')
plt.axvline(median_income, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_income:,.0f}')

plt.legend()

plt.savefig('income_histogram.png')
plt.close()

print(f"Mean Income: ${mean_income:,.2f}")
print(f"Median Income: ${median_income:,.2f}")
print(f"Skewness: {df_cleaned['Income'].skew():.2f}")
print(f"Kurtosis: {df_cleaned['Income'].kurtosis():.2f}")


# Question 5


df = pd.read_csv('censusdata/acs2015_census_tract_data (1).csv')

df_cleaned = df[['Income', 'IncomePerCap']].dropna()

correlation = df_cleaned['Income'].corr(df_cleaned['IncomePerCap'])
print(f"Correlation between Income and IncomePerCap: {correlation}")

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Income'], df_cleaned['IncomePerCap'], color='blue', alpha=0.5)

plt.title('Income vs IncomePerCap', fontsize=15)
plt.xlabel('Income', fontsize=12)
plt.ylabel('IncomePerCap', fontsize=12)

plt.savefig('income_scatter.png')

plt.show()

# Question 6

df = pd.read_csv('censusdata/acs2015_census_tract_data (1).csv')

df_cleaned = df[['Professional', 'Income']].dropna()

correlation = df_cleaned['Professional'].corr(df_cleaned['Income'])
print(f"Correlation between Professional percentage and Income: {correlation}")

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Professional'], df_cleaned['Income'], color='green', alpha=0.5)

plt.title('Professional Employment vs Income', fontsize=15)
plt.xlabel('Percentage of Professional Workers', fontsize=12)
plt.ylabel('Income', fontsize=12)

plt.savefig('professional_income_scatter.png')

plt.show()

# Question 7

df = pd.read_csv('censusdata/acs2015_census_tract_data (1).csv')

df_cleaned: DataFrame = df.dropna(subset=['Unemployment', 'Employed', 'Income', 'Professional', 'WorkAtHome', 'TotalPop'])

# a) Calculate UnemploymentRate and analyze its relationship with Income
df_cleaned['UnemploymentRate'] = (df_cleaned['Unemployment'] / df_cleaned['Employed']) * 100

# Calculate correlation between UnemploymentRate and Income
correlation_unemployment_income = df_cleaned['UnemploymentRate'].corr(df_cleaned['Income'])
print(f"Correlation between Unemployment Rate and Income: {correlation_unemployment_income}")

# Create scatter plot: Unemployment Rate vs Income
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['UnemploymentRate'], df_cleaned['Income'], color='purple', alpha=0.5)
plt.title('Unemployment Rate vs Income', fontsize=15)
plt.xlabel('Unemployment Rate (%)', fontsize=12)
plt.ylabel('Income', fontsize=12)
plt.savefig('unemployment_vs_income_scatter.png')
plt.show()

# b)
state_unemployment = df_cleaned.groupby('State')['UnemploymentRate'].mean().sort_values(ascending=False)

top_10_unemployment_states = state_unemployment.head(10)
print("Top 10 states with the highest unemployment rates:")
print(top_10_unemployment_states)

plt.figure(figsize=(12, 6))
top_10_unemployment_states.plot(kind='bar', color='red', edgecolor='black')
plt.title('Top 10 States with Highest Unemployment Rates', fontsize=15)
plt.xlabel('State', fontsize=12)
plt.ylabel('Average Unemployment Rate (%)', fontsize=12)
plt.xticks(rotation=45)
plt.savefig('top10_unemployment_by_state.png')
plt.show()

# c)
correlation_unemployment_professional = df_cleaned['UnemploymentRate'].corr(df_cleaned['Professional'])
print(f"Correlation between Unemployment Rate and Professional Percentage: {correlation_unemployment_professional}")

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Professional'], df_cleaned['UnemploymentRate'], color='green', alpha=0.5)
plt.title('Professional Workers vs Unemployment Rate', fontsize=15)
plt.xlabel('Percentage of Professional Workers (%)', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.savefig('unemployment_vs_professional_scatter.png')
plt.show()

# d)
df_cleaned['WorkAtHomePercentage'] = (df_cleaned['WorkAtHome'] / df_cleaned['TotalPop']) * 100
correlation_unemployment_work_at_home = df_cleaned['UnemploymentRate'].corr(df_cleaned['WorkAtHomePercentage'])
print(f"Correlation between Unemployment Rate and Work At Home Percentage: {correlation_unemployment_work_at_home}")

plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['WorkAtHomePercentage'], df_cleaned['UnemploymentRate'], color='blue', alpha=0.5)
plt.title('Work From Home Percentage vs Unemployment Rate', fontsize=15)
plt.xlabel('Work From Home Percentage (%)', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.savefig('unemployment_vs_work_from_home_scatter.png')
plt.show()

# Question 8


df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")


plt.figure(figsize=(10, 6))
sns.regplot(x='Poverty', y='Unemployment', data=df, scatter_kws={'alpha':0.5})
plt.title('Poverty vs Unemployment')
plt.xlabel('Poverty Rate')
plt.ylabel('Unemployment Rate')


plt.savefig('poverty_unemployment_scatter.png')
plt.close()

print("Scatter plot saved as 'poverty_unemployment_scatter.png'")

correlation = df['Poverty'].corr(df['Unemployment'])
print(f"Correlation between Poverty and Unemployment: {correlation:.4f}")

# Question 9

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")

df['Income_Quintile'] = pd.qcut(df['Income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

transport_cols = ['Drive', 'Transit', 'Walk', 'WorkAtHome']
transport_means = df.groupby('Income_Quintile')[transport_cols].mean()

fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(5)
for col in transport_cols:
    ax.bar(transport_means.index, transport_means[col], bottom=bottom, label=col)
    bottom += transport_means[col]

ax.set_title('Transportation Methods by Income Quintile')
ax.set_xlabel('Income Quintile')
ax.set_ylabel('Percentage')
ax.legend(title='Transportation Method')

for i, quintile in enumerate(transport_means.index):
    total = 0
    for col in transport_cols:
        value = transport_means.loc[quintile, col]
        ax.text(i, total + value/2, f'{value:.1f}%', ha='center', va='center')
        total += value

plt.tight_layout()
plt.savefig('transportation_by_income.png')
plt.close()

print("Stacked bar chart saved as 'transportation_by_income.png'")

print(transport_means)


# Question 10


df = pd.read_csv("censusdata/acs2015_census_tract_data (1).csv")


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Poverty', y='ChildPoverty', data=df, alpha=0.5)
plt.title('Overall Poverty vs Child Poverty')
plt.xlabel('Overall Poverty Rate (%)')
plt.ylabel('Child Poverty Rate (%)')


plt.plot([0, 100], [0, 100], linestyle='--', color='red', alpha=0.5)


plt.savefig('poverty_comparison.png')
plt.close()

print("Scatter plot saved as 'poverty_comparison.png'")


correlation = df['Poverty'].corr(df['ChildPoverty'])
print(f"Correlation between Overall Poverty and Child Poverty: {correlation:.4f}")


df['Poverty_Diff'] = df['ChildPoverty'] - df['Poverty']
interesting_cases = df[abs(df['Poverty_Diff']) > 20].sort_values('Poverty_Diff', ascending=False)
print("\nInteresting cases (where child poverty differs significantly from overall poverty):")
print(interesting_cases[['State', 'County', 'Poverty', 'ChildPoverty', 'Poverty_Diff']].head())