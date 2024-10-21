## Credit Card Fraud Detection

This notebook focuses on classifying credit card transactions to identify fraudulent ones, using the known dataset available on Kaggle 
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This binary classification problem is characterize by an highly unbalanced 
dataset since out of 284,807 transactions, only 492 are fraudulent. Following an exploratory data analysis (EDA), the dataset is split 
into training and testing sets. The final model chosen will consist of a multiple ensemble of bagged algorithms. The out-of-sample 
predictions of the trained model yield the following statistics:

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**         | 0.999631  |
| **Balanced Accuracy**| 0.923417  |
| **ROC AUC**          | 0.983186  |
| **Log Loss**         | 0.002136  |
| **Precision**        | 0.932584  |
| **Recall**           | 0.846939  |
| **F1 Score**         | 0.887701  |
| **Kappa**            | 0.887516  |
| **Average Precision**| 0.790105  |




# IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# Plots the deposit faciity rate and the marginal lending facility rate.
def costant_TS(DF):
    Columns = DF.columns
    DF.columns = ["col1", "col2"]
    values = [i for i in DF["col1"]]
    dates = [i for i in DF["col2"]]
    dates.append(pd.to_datetime("27 May 2024")) #pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
    TS = pd.Series(values[0], pd.date_range(start =pd.to_datetime(dates[0]), end = pd.to_datetime(dates[1])))

    for i in range(1,len(DF)):
        j = i + 1
        TS = pd.concat([TS, pd.Series(values[i], pd.date_range(start =pd.to_datetime(dates[i]), end = pd.to_datetime(dates[j])))])
    
    TS = pd.DataFrame(data = TS,
                      columns = [Columns[0]])
    return(TS)

data = {"Deposit Facility": [-0.003, -0.004, -0.005, 0,
                             0.0075, 0.0150, 0.0200, 0.0250, 0.0300, 0.0325, 0.0350, 0.0375, 0.0400],
        "Date": [pd.to_datetime("09 Dec. 2015"), pd.to_datetime("16 Mar. 2016"), pd.to_datetime("18 Sep. 2016"), pd.to_datetime("27 Jul. 2022"), 
                 pd.to_datetime("14 Sep. 2022"), pd.to_datetime("2 Nov. 2022"), pd.to_datetime("21 Dec. 2022"), pd.to_datetime("8 Feb. 2023"), pd.to_datetime("22 Mar. 2023"), pd.to_datetime("10 May 2023"), pd.to_datetime("21 Jun. 2023"), pd.to_datetime("2 Aug. 2023"), pd.to_datetime("20 Sep. 2023")]}

DF = pd.DataFrame(data)
DF

data = {"Marginal lending facility": [0.003, 0.0025, 0.0025, 0.0075,
                                      0.015, 0.0225, 0.0275, 0.0325, 0.0375, 0.0400, 0.0425, 0.045, 0.0475],
        "Date": [pd.to_datetime("09 Dec. 2015"), pd.to_datetime("16 Mar. 2016"), pd.to_datetime("18 Sep. 2016"), pd.to_datetime("27 Jul. 2022"), 
                 pd.to_datetime("14 Sep. 2022"), pd.to_datetime("2 Nov. 2022"), pd.to_datetime("21 Dec. 2022"), pd.to_datetime("8 Feb. 2023"), pd.to_datetime("22 Mar. 2023"), pd.to_datetime("10 May 2023"), pd.to_datetime("21 Jun. 2023"), pd.to_datetime("2 Aug. 2023"), pd.to_datetime("20 Sep. 2023")]}

DF2 = pd.DataFrame(data)

DF3 = pd.concat([costant_TS(DF), costant_TS(DF2)], axis = 1)

# Plot all the time series with different maturities.
TS_for_PCA_EUR = TS[TS["Indicator"] == "ESTR"]
TS_for_PCA_EUR = TS_for_PCA_EUR[["Rate","Maturity"]]
pivot_table = TS_for_PCA_EUR.pivot_table(values='Rate', index=TS_for_PCA_EUR.index, columns='Maturity')
plt.plot(pivot_table)

# Plot the yield curves with a varying color for each maturity.
sorted_maturities = sorted(pivot_table.columns, key=lambda x: int(x[:-1]) if x[-1] in ['M', 'Y'] else int(x))
pivot_table_sorted = pivot_table[sorted_maturities]
dates_numeric = [i.timestamp() for i in pivot_table_sorted.index.tolist()]

norm = Normalize(vmin=min(dates_numeric), vmax=max(dates_numeric))
cmap = plt.get_cmap('viridis')
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig, ax = plt.subplots(figsize=(12, 8))
for date in pivot_table_sorted.index:
    date_numeric = date.timestamp()
    ax.plot(sorted_maturities, pivot_table_sorted.loc[date], color=cmap(norm(date_numeric)))

cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(min(dates_numeric), max(dates_numeric), 10))
cbar.ax.set_yticklabels([str(pd.to_datetime(tick, unit='s').date()) for tick in cbar.get_ticks()])

ax.set_title('Yield Curves Over Time (ESTR)')
ax.set_xlabel('Maturities')
ax.set_ylabel('Interest Rates')

xticks = np.linspace(0, len(sorted_maturities) - 1, 10).astype(int)
ax.set_xticks(xticks)
ax.set_xticklabels([sorted_maturities[i] for i in xticks], rotation=45)
ax.grid(True)
plt.show()

# Perform PCA and plot the three principal components.
standardized_data = StandardScaler().fit_transform(pivot_table_sorted)
N_com = 3
pca = PCA(n_components=N_com)
principal_components = pca.fit_transform(standardized_data)
PCA_df = pd.DataFrame(data=principal_components, columns=["PC" + str(i) for i in range(1, N_com + 1)])

# Plot 1
PCA_df.plot(figsize=(10, 5))
plt.title('Principal Components Over Time')
plt.xlabel('Time')
plt.ylabel('Principal Component Value')
plt.show()

# Plot 2
plt.figure(figsize=(10, 5))
plt.bar(range(1, N_com + 1), pca.explained_variance_ratio_, align="center")
plt.xticks(range(1, N_com + 1), ["PC" + str(i) for i in range(1, N_com + 1)])
plt.title("Variance Explained by Each Component")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.show()

# Plot 3
factor_load = pd.DataFrame(pca.components_.T, index=pivot_table_sorted.columns, columns=[f"PC{i}" for i in range(1, N_com+1)])
factor_load.plot(title="Factor Loadings", figsize=(10, 5))
plt.ylabel("Weight")
plt.xlabel("Maturity")
plt.show()

# Perform PCA to all of its components (i.e. no loss of information) and set the PC1 to 0 to exclude level changes.
scaler = StandardScaler()
standardized_data = scaler.fit_transform(pivot_table_sorted)

N_com = len(pivot_table_sorted.columns) # 257
pca = PCA(n_components=N_com)
principal_components = pca.fit_transform(standardized_data)

PCA_df = pd.DataFrame(data=principal_components, columns=["PC" + str(i) for i in range(1, N_com + 1)])
PCA_df_without_PC1 = PCA_df.copy()
PCA_df_without_PC1.iloc[:, 0] = 0  # Setting the PC1 to 0 I exclude the level changes.

# Reconstruct the initial data with the new set of principal components.
reconstructed_data_standardized = pca.inverse_transform(PCA_df_without_PC1)
reconstructed_data = scaler.inverse_transform(reconstructed_data_standardized)
reconstructed_df = pd.DataFrame(reconstructed_data, index=pivot_table_sorted.index, columns=pivot_table_sorted.columns)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_df, label='Reconstructed Series (excluding 1st PC)')
plt.title('Time series with PC1 set to 0')
plt.xlabel('Time')
plt.ylabel('Original Scale')
plt.show()

# Plot the two graphs with the evolutions of the Yield curves with and without the PC1.
sorted_maturities_pivot = sorted(pivot_table.columns, key=lambda x: int(x[:-1]) if x[-1] in ['M', 'Y'] else int(x))
sorted_maturities_reconstructed = sorted(reconstructed_df.columns, key=lambda x: int(x[:-1]) if x[-1] in ['M', 'Y'] else int(x))

pivot_table_sorted = pivot_table[sorted_maturities_pivot]
reconstructed_df_sorted = reconstructed_df[sorted_maturities_reconstructed]
dates_numeric = [i.timestamp() for i in pivot_table_sorted.index.tolist()]

norm = Normalize(vmin=min(dates_numeric), vmax=max(dates_numeric))
cmap = plt.get_cmap('viridis')
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)


for date in pivot_table_sorted.index:
    date_numeric = date.timestamp()
    ax1.plot(sorted_maturities_pivot, pivot_table_sorted.loc[date], color=cmap(norm(date_numeric)), alpha=0.6)

for date in reconstructed_df_sorted.index:
    date_numeric = date.timestamp()
    ax2.plot(sorted_maturities_reconstructed, reconstructed_df_sorted.loc[date], color=cmap(norm(date_numeric)), alpha=0.6)

cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
cbar.ax.set_xticklabels([str(pd.to_datetime(tick, unit='s').date()) for tick in cbar.get_ticks()], rotation=45)
cbar.set_label('Dates')


ax1.set_title('Original Yield Curves Over Time')
ax1.set_xlabel('Maturities')
ax1.set_ylabel('Interest Rates')
ax2.set_title('Yield Curves Over Time with PC1 set to 0')
ax2.set_xlabel('Maturities')
xticks = np.linspace(0, len(sorted_maturities_pivot) - 1, 10).astype(int)
ax1.set_xticks(xticks)
ax1.set_xticklabels([sorted_maturities_pivot[i] for i in xticks], rotation=45)
ax2.set_xticks(xticks)
ax2.set_xticklabels([sorted_maturities_pivot[i] for i in xticks], rotation=45)
ax1.grid(True)
ax2.grid(True)

plt.show()

# Perform PCA to all of its components (i.e. no loss of information) and set the PC1 to 0 to exclude level changes.
scaler = StandardScaler()
standardized_data = scaler.fit_transform(pivot_table_sorted)

N_com = 1 #len(pivot_table_sorted.columns)
pca = PCA(n_components=N_com)
principal_components = pca.fit_transform(standardized_data)

PCA_df = pd.DataFrame(data=principal_components, columns=["PC" + str(i) for i in range(1, N_com + 1)])
PCA_df_without_PC1 = PCA_df.copy()
#PCA_df_without_PC1.iloc[:, 0] = 0  # Setting the PC1 to 0 I exclude the level changes.

# Reconstruct the initial data with the new set of principal components.
reconstructed_data_standardized = pca.inverse_transform(PCA_df_without_PC1)
reconstructed_data = scaler.inverse_transform(reconstructed_data_standardized)
reconstructed_df = pd.DataFrame(reconstructed_data, index=pivot_table_sorted.index, columns=pivot_table_sorted.columns)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_df, label='Reconstructed Series (excluding 1st PC)')
plt.title('Time series with only PC1')
plt.xlabel('Time')
plt.ylabel('Original Scale')
plt.show()
