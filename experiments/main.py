# %% [markdown]
# 

# %% [markdown]
# Prediction Questions Your Model Can Answer:
# 1. “How many tons of product will a warehouse produce given its zone, worker count, and distance from the hub?”
# 
# 2. “What is the estimated product weight output if the warehouse is located in a rural zone with mid-size capacity?”
# 
# 3. “How much product (in tons) should we expect from a warehouse with 30 workers and an A+ government certificate?”
# 
# 4. “If a warehouse is in Zone 4 and 25 km from the hub, how much product weight can we predict?”
# 
# 5. “Based on warehouse characteristics, what will be the total product_wg_ton this facility is likely to generate?”

# %% [markdown]
# 

# %%
print("hellow world")

# %%
import warnings
warnings.filterwarnings("ignore")


# %%
%pip install pandas numpy matplotlib seaborn


# %%
%pip install plotly


# %%
%pip install scikit-learn


# %%
import pandas as pd         # For data handling
import numpy as np          # For numerical operations
import matplotlib.pyplot as plt   # For plots
import seaborn as sns       # For advanced visualizations


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %%
df = pd.read_csv("data.csv")

# %%
df.head(10)

# %%
# Show basic structure
print(df.shape)



# %%
print(df.columns)

# %%

print(df.info())

# %%
df.isnull().sum()

# %% [markdown]
# 

# %% [markdown]
# ### Zero Variance

# %%
variances = df.var(numeric_only=True)

variances_sorted = variances.sort_values(ascending=True)

for feature, var in variances_sorted.items():
    print(f"{feature:<30} {var:.5f}")


# %% [markdown]
# 

# %% [markdown]
# ### Categorical Cardinality Analysis

# %%
for feature in df.select_dtypes('object'):
    print(feature, df[feature].nunique())


# %%
df.columns

# %% [markdown]
# ### Outlier Check

# %%
df.describe()

# %%
df.describe().loc['max'].sort_values(ascending=False)


# %% [markdown]
# We compare mean vs 50% (median) to quickly check if the data is skewed or has outliers. If both values are close → data is balanced ✅. But if the mean is much higher or lower than median, it means some extreme values are pulling the average — that's a red flag for outliers 🚨. It’s just a smart, early warning before plotting or cleaning.

# %%
df.describe().loc[['mean','50%'],:]


# %% [markdown]
# Purpose: To check skewness in numeric columns.
# 
# Why?
# If mean ≈ median → distribution is likely symmetric (normal or near-normal).
# 
# If mean ≫ median → right-skewed (possible high outliers).
# 
# If mean ≪ median → left-skewed (possible low outliers).

# %%
for feature in df.select_dtypes(include=['int', 'float']):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outlier_count = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
    outlier_percentage = (outlier_count / len(df)) * 100

    print(f"{feature:<30} Outliers: {outlier_count:<5} ({outlier_percentage:.2f}%)")


# %% [markdown]
# In the outlier detection process, we're using statistical methods like the IQR rule to identify columns that might contain extreme values. But this method is designed for continuous numeric variables that have a natural range and spread. When we include binary columns like flood_impacted or flood_proof, which only have values like 0 and 1, the IQR logic fails — it falsely identifies the less frequent value (like 1) as an outlier simply because it’s rare.
# 
# So, we remove (skip) these binary columns from the outlier detection loop, not because they are bad or useless, but because the outlier logic doesn't make sense for them. They are still kept in the dataset and will be used later during model training, because binary features often carry strong predictive power in classification or regression tasks.
# 
# In short: we’re not removing these columns from the data, we’re just removing them from outlier logic to avoid incorrect handling. This keeps the dataset statistically clean without losing important information.
# 
# 
# 
# 
# 
# 
# 
# 

# %%
# Recreate the list first (if you’re in a new cell)
outlier_containing_features = ['transport_issue_l1y', 'Competitor_in_mkt', 'retail_shop_num', 'workers_num', 'flood_impacted', 'flood_proof']

# Now remove binary columns from it
outlier_containing_features.remove('flood_proof')
outlier_containing_features.remove('flood_impacted')

# Check updated list
print(outlier_containing_features)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.subplots(figsize=(12, 12))

for i, feature in enumerate(outlier_containing_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(df[feature])
    plt.title(feature)
    plt.xlabel("")
    
plt.tight_layout()
plt.show()


# %%
df_treated = df.copy()

for feature in outlier_containing_features:
    # Ensure the column is float to avoid dtype warning
    df_treated[feature] = df_treated[feature].astype(float)

    # Calculate IQR bounds
    Q1 = df_treated[feature].quantile(0.25)
    Q3 = df_treated[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Apply capping
    df_treated.loc[df_treated[feature] < lower, feature] = lower
    df_treated.loc[df_treated[feature] > upper, feature] = upper


# %%
df_treated.describe()

# %%
binary_cols = ['flood_impacted', 'flood_proof']

for feature in df_treated.select_dtypes(include=['int', 'float']):
    if feature in binary_cols:
        continue  # ❌ Skip binary columns

    Q1 = df_treated[feature].quantile(0.25)
    Q3 = df_treated[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers in df_treated
    outlier_count = ((df_treated[feature] < lower_bound) | (df_treated[feature] > upper_bound)).sum()
    outlier_percentage = (outlier_count / len(df_treated)) * 100

    print(f"{feature:<30} Outliers: {outlier_count:<5} ({outlier_percentage:.2f}%)")


# %%
df_treated.head()

# %%
print(df_treated.columns)


# %%
print("Total columns:", len(df_treated.columns))

# %%
df_treated.head(10)

# %%
df_treated.columns

# %%
df_treated.info()

# %%
df_treated = df_treated.drop(columns=['Unnamed: 0'])

# %%
df_treated.columns

# %%
pip install ipython nbformat


# %%
df_treated.head()

# %%
numeric_cols = data.select_dtypes(include=['number']).columns
for col in numeric_cols:
    sns.kdeplot(data=data, x=col)
    plt.title(f"KDE Plot for {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()

    

# %%
df_treated.columns

# %%
np.log(df_treated['product_wg_ton'])

# %%
sns.kdeplot(x=np.log(df_treated['product_wg_ton']))

# %%
from scipy.stats import boxcox
df_treated['product_wg_ton'], lmd= boxcox(df_treated['product_wg_ton'])

# %%
sns.kdeplot(x= df_treated['product_wg_ton'])

# %% [markdown]
# ### Start Buidling model

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# %%
from sklearn.model_selection import train_test_split

# %%
selected_columns = [
    'Location_type',
    'WH_capacity_size',
    'zone',
    'WH_regional_zone',
    'wh_owner_type',
    'approved_wh_govt_certificate'
]

for col in selected_columns:
    unique_vals = df_treated[col].nunique()
    
    print(f"{col}: {unique_vals} unique values")


# %% [markdown]
# ### Let’s apply One-Hot Encoding using pandas.get_dummies() on your 6 categorical columns, and store the result in a new DataFrame called df_dummies.

# %%

# Create df_dummies with One-Hot Encoding
df_dummies = pd.get_dummies(
    df_treated,
    columns=[
        'Location_type',
        'WH_capacity_size',
        'zone',
        'WH_regional_zone',
        'wh_owner_type',
        'approved_wh_govt_certificate'
    ],
    drop_first=True  # Avoid multicollinearity
)

# Optional: check new shape
print("Shape before encoding:", df_treated.shape)
print("Shape after encoding :", df_dummies.shape)


# %%
df_dummies = df_dummies.astype({col: int for col in df_dummies.select_dtypes('bool').columns})


# %%
df_dummies.info()

# %%
X = df_dummies.drop('product_wg_ton', axis=1)
y = df_dummies['product_wg_ton']

X.info()

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80/20 split
    random_state=42       # for reproducibility
)

# %%
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluation Function
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return r2, mae, mse

# Models Dictionary
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(max_iter=1000),
    'Ridge': Ridge(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100)
}

# Evaluate Each Model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2, mae, mse = evaluate_model(y_test, y_pred)

    print(f"\n{name} Performance")
    print(f"R2 Score : {r2*100:.2f}")
    print(f"MAE      : {mae:.2f}")
    print(f"MSE      : {mse:.2f}")
    print("-" * 30)


# %% [markdown]
# ### LinearRegression Performance
# R2 Score : 97.97
# MAE      : 19.93
# MSE      : 710.10

# %%
final_model = LinearRegression()
final_model.fit(X, y)


# %%
import pickle

with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)


# %%
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("Final R²:", r2_score(y_test, y_pred))
print("Final MAE:", mean_absolute_error(y_test, y_pred))
print("Final MSE:", mean_squared_error(y_test, y_pred))


# %% [markdown]
# Final Model Evaluation Summary
# 1. R² Score: 0.9921 → Excellent fit (explains 99.2% of variance in y_test)
# 
# 2. MAE: 11.63 → On average, predictions are off by ~11.6 units
# 
# 3. MSE: 275.30 → Squared average error (for detecting larger deviations)

# %%
X.head(10)

# %%


# Replace with real values matching the 30 feature inputs
new_input = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 42, 0, 0, 1, 0.0, 3.0, 2.0, 6000, 1, 0, 0, 50, 0, 0, 1, 1, 1, 3, 1, 0, 0, 1]])

predicted_output = model.predict(new_input)
print("Predicted Product Weight:", predicted_output[0])


# %%
print(X.columns.tolist())


# %% [markdown]
# ### Sanity Check Output

# %%
print("Min:", y.min())
print("Max:", y.max())
print("Mean:", y.mean())


# %% [markdown]
# ## 🔚 Final Conclusion – Product Weight Prediction (Linear Regression)
# 
# I developed a machine learning model to predict product weight (`product_wg_ton`) using various warehouse and logistics features.
# 
# ### 🔍 Key Results:
# - **Model Chosen**: Linear Regression
# - **Train-Test R² Score**: 97.97% → Excellent fit
# - **Final Prediction R²**: 99.21% → Very high accuracy
# - **Mean Absolute Error (MAE)**: ~11.63 units
# - **Mean Squared Error (MSE)**: ~275.30
# 
# ### ✅ Why This Model?
# - Linear Regression gave highly interpretable results
# - Performance is stable across the test set
# - Low error rates, high generalization
# 
# ### 📊 Prediction Output Range:
# - **Actual Values Range**: 145.7 to 1001.0
# - **Predicted Sample Output**: 233.82 → ✅ Within range
# 
# ### 📦 Business Use:
# - Enables real-time prediction of product weight for supply chain planning
# - Assists in optimizing delivery loads and warehouse operations
# 
# ---
# 

# %% [markdown]
# 

# %%
def main():
    print("Hellow from Project_Supplychain!")

if __name__== "__main__":
    main()


# %%



