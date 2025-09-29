# Data Cleaning with AI Support

## Student Information
- Name: GIO KIEFER A. SANCHEZ
- Course Year: BSCS 4
- Date: 2025-09-29

## Dataset
- Source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- Name: House Prices - Advanced Regression Techniques

## Issues found
- Missing values: [HIGH MISSINGNESS] PoolQC, MiscFeature, Alley, Fence
                  [MODERATE MISSINGNESS] Lot Frontage
                  [VALUES DEPENDENT ON A COLUMN FOR THEIR MISSINGNESS (if val=0, DNE)]
                        Dependent on TotalBsmtSF - BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, BsmtFullBath, BsmtHalfBath
                        Dependent on GarageArea - GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond 
                        Dependent on Fireplaces - FireplaceQu
                        Dependent on PoolArea - PoolQC
                        Dependent on MasVnrArea - MasVnrType
                        Dependent on MiscVal - MiscFeature
- Duplicates: None (Based on ID-inclusive and ID-agnostic checks)
- Inconsistencies: [MSZoning] "C (all)" must be standardized to "C"
                   [MSZoning] "Ideally, "Twnhs" should be subclassified into "TwnhsE" and "TwnhsI." Keep "Twnhs" as information is lacking to perform this split.
                   [Exterior2nd] Contains unexpected classes when crossreferenced with data_description.txt: "CmentBd", "Wd Shng", and "Brk Cmn". As these classes exist in Exterior1st under different names ("CemntBd", "WdShing", "BrkCmn"), we will standardize/correct these class names. 

## Cleaning steps
1. Missing values: [HIGH MISSINGNESS] Drop columns
                   [MODERATE MISSINGNESS] Impute with median
                   [VALUES DEPENDENT ON A COLUMN FOR THEIR MISSINGNESS] Set to NA where corresponding count column is 0.
2. Duplicates: No Action Needed
3. Inconsistencies: [MSZoning] Standardize by replacing "C (all)" with "C".
                    [Exterior2nd] Correct typos and Inconsistencies.
                            -> Replace "CmentBd" with "CemntBd".
                            -> Standardize "Wd Shng" to "WdShing"
                            -> Standardize "Brk Cmn" to "BrkCmn"
4. Outliers: After identifying the chosen columns (in SalePrice, LotArea, and GrLivArea) as positively skewed, LOG TRANSFORM was chosen as an appropriate method to make the distributions more normal (symmetrical)

## AI prompts used

- Prompt 1:
"""
<data_description.txt>
    [contents of data_description.txt go here.]
</data_description.txt>
given this data_description.txt, I would like to improve this block of code:
<code>
# next, let's review the contents of our CSV.
desc = df.describe()
print("Statistics:\n")
print(desc)
# let's explore this further by reviewing the freq/class in each column.
print("\n\nValue counts per column:\n")
for col in df.columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False).head(20))
</code>
I believe that we can (1) express the categorical columns as an array of values and (2) from the dataset df, detect for anomalous data (like classes that shouldn't be there, data that should necessarily be present but are absent, outliers, and so on.)
Provide the improved version of this block of code. Do so without being overcomplicated and sticking to the original structure of our code. Keep the code clean and readable. This will be going in a python notebook, but I would like to fit this section into one cell (the part highlighting the use of df.describe). I would also like to compare expected classes (as seen in data_description.txt) and look for unexpected content. you're allowed to explicitly write them down as arrays in the code, as I cannot include the data_description.txt file into my repository.
"""

- Generated code:
"""
# Define expected categorical values based on data description
EXPECTED_CATEGORIES = {
    'MSSubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
    'Street': ['Grvl', 'Pave'],
    'Alley': ['Grvl', 'Pave', 'NA'],
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
    'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
    'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
    'LandSlope': ['Gtl', 'Mod', 'Sev'],
    'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 
                    'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 
                    'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 
                    'Somerst', 'StoneBr', 'Timber', 'Veenker'],
    'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
    'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
    'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
    'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
    'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 
                   'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 
                   'VinylSd', 'Wd Sdng', 'WdShing'],
    'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 
                   'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 
                   'VinylSd', 'Wd Sdng', 'WdShing'],
    'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
    'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
    'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'],
    'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
    'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'PavedDrive': ['Y', 'P', 'N'],
    'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
    'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
    'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
    'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
}

# Review basic statistics
desc = df.describe()
print("Statistics:\n")
print(desc)

# Check for missing values
print("\n\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Review value counts and detect anomalies
print("\n\nValue counts and data quality checks per column:\n")
for col in df.columns:
    print(f"\n--- {col} ---")
    value_counts = df[col].value_counts(dropna=False).head(20)
    print(value_counts)
    
    # Check for unexpected categorical values
    if col in EXPECTED_CATEGORIES:
        unique_vals = set(df[col].dropna().unique())
        expected_vals = set(EXPECTED_CATEGORIES[col])
        unexpected_vals = unique_vals - expected_vals
        
        if unexpected_vals:
            print(f"⚠️  UNEXPECTED VALUES: {unexpected_vals}")
    
    # Check for null representations
    null_representations = ['NA', 'None', 'null', 'NULL', '', ' ', 0]
    found_nulls = [repr for repr in null_representations if repr in df[col].values]
    if found_nulls:
        print(f"📝 NULL REPRESENTATIONS: {found_nulls}")
        
    # Check for zeros in areas/sizes where they might be suspicious
    area_columns = ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if col in area_columns and (df[col] == 0).any():
        zero_count = (df[col] == 0).sum()
        print(f"⚠️  ZERO VALUES: {zero_count} records")
    
    # Check for negative values (shouldn't exist for most columns)
    if df[col].dtype in ['int64', 'float64'] and (df[col] < 0).any():
        negative_count = (df[col] < 0).sum()
        print(f"⚠️  NEGATIVE VALUES: {negative_count} records")
        
    # Check for outliers using IQR
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        if not outliers.empty:
            print(f"⚠️  OUTLIERS: {len(outliers)} values outside [{lower:.2f}, {upper:.2f}]")
"""


- Prompt 2:
"""
Reviewing the data, the following issues that need to be cleaned arise:

1. MSZoning codes are all single letters/combinations without parenthesis. We will replace "C (all)" with "C" for consistency.
2. In data_description text, Twnhs should be classified under TnwhsE or TwhnsI.
   With no supporting information from the given columns, we will take a conservative approach
   and express all townhouses with Twnhs due to the lack of additional knowledge.
3. "CmentBd", "Wd Shng", and "Brk Cmn" are all unexpected classes in Exterior2nd.
   However, "CmentBd" is obviously a typo of "CemntBd" present in both Exterior2nd and Exterior1st.
   Also, seeing as similar categories are present in Exterior1st (as WdShing and BrkCmn), we must standardize Exterior2nd.
4. For area/size columns (these are the ones missing values),
5. There are columns which affect others, meaning that if set to 0, other columns should be set to NA.
   for example,  if no TotalBsmtSF, then BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
   should be NA (NO BASEMENT, these values would not make sense)
   these are the relationships we have identified
   column to check if 0 -> column/s to set as NA
      TotalBsmtSF -> BsmtQual, BsmtCond, BsmtExposure, 
                     BsmtFinType1,  BsmtFinType1,
                     BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, BsmtFullBath,
                     BsmtHalfBath
      GarageArea -> GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond
      Fireplaces -> FireplaceQu
      PoolArea -> PoolQC
      MasVnrArea -> MasVnrType
      MiscVal -> MiscFeature
6. Duplicate IDs (or duplicated content ignoring IDs) don't need to be addressed anymore.
   We have shown that these do not exist earlier.
7. Remove outliers to normalize data in the following columns: SalePrice, LotArea, GrLivArea
   While they may be real, they are outliers nevertheless. We can normalize them using a log scale.
   However, i'd like to check if these values skewed enough to justify log transform.
   After testing, I can confirm that it's appropriate!
8. Handle missing data accordingly:
   a. DROP 'PoolQC','MiscFeature','Alley','Fence'. (>90% missing, basically useless)
   b. Moderately empty numeric columns (LotFrontage) will be filled with median
   c. Handle low missingness with the NA convention we talked about above (related columns)

My data can be found in ../data/raw_dataset.csv. Given this information, take the data from my raw_dataset. Write a code to perform the cleaning given my instructions above, and export it as ../data/cleaned_dataset.csv.
"""

- Generated Code:

import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv('../data/raw_dataset.csv')

# 1. Standardize categorical values
df['MSZoning'] = df['MSZoning'].replace('C (all)', 'C')
df['Exterior2nd'] = df['Exterior2nd'].replace({
    'CmentBd': 'CemntBd',
    'Wd Shng': 'WdShing', 
    'Brk Cmn': 'BrkCmn'
})

# 2. Handle conditional NA values based on area/size columns
# Basement-related columns
basement_condition = (df['TotalBsmtSF'] == 0) | (df['TotalBsmtSF'].isna())
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
df.loc[basement_condition, basement_cols] = 'NA'

# Garage-related columns  
garage_condition = (df['GarageArea'] == 0) | (df['GarageArea'].isna())
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df.loc[garage_condition, garage_cols] = 'NA'

# Other conditional columns
df.loc[df['Fireplaces'] == 0, 'FireplaceQu'] = 'NA'
df.loc[df['PoolArea'] == 0, 'PoolQC'] = 'NA'
df.loc[df['MiscVal'] == 0, 'MiscFeature'] = 'NA'

# Masonry veneer - use 'None' as specified in data description
df.loc[df['MasVnrArea'] == 0, 'MasVnrType'] = 'None'

# 3. Handle missing data
# Drop columns with >90% missing
df = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])

# Fill moderate missingness with median
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# 4. Remove outliers using log transformation and IQR
skewed_columns = ['SalePrice', 'LotArea', 'GrLivArea']

# Apply log transformation
for col in skewed_columns:
    df[f'log_{col}'] = np.log1p(df[col])

# Calculate IQR bounds for log-transformed values
bounds = {}
for col in skewed_columns:
    log_col = f'log_{col}'
    Q1 = df[log_col].quantile(0.25)
    Q3 = df[log_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bounds[log_col] = (lower_bound, upper_bound)

# Filter outliers across all three columns simultaneously
mask = pd.Series(True, index=df.index)
for col, (lower, upper) in bounds.items():
    mask &= (df[col] >= lower) & (df[col] <= upper)

df = df[mask]

# Scale back from log transform and format as whole numbers
for col in skewed_columns:
    log_col = f'log_{col}'
    df[col] = np.expm1(df[log_col]).round().astype(int)
    df = df.drop(columns=[log_col])

# 5. Handle MasVnrType None values
df['MasVnrType'] = df['MasVnrType'].fillna('None')

# Export cleaned dataset
df.to_csv('../data/cleaned_dataset.csv', index=False)

print("Data cleaning completed successfully!")
print(f"Final dataset shape: {df.shape}")

print(f"\nCleaning Summary:")
print(f"- Original rows: {pd.read_csv('../data/raw_dataset.csv').shape[0]}")
print(f"- Final rows: {df.shape[0]}")


## Results
- Rows before: 1460
- Rows after: 1303

Video: link
        
Quick Notebook template (click to expand)
# Data Cleaning Notebook Template

## 1. Import Libraries
import pandas as pd
import numpy as np

## 2. Load Dataset
df = pd.read_csv("../data/raw_dataset.csv")
df.head()

## 3. Initial Exploration
df.info()
df.describe()

## 4. Handle Missing Values
# Example: fill numeric NaNs with median
df['column'] = df['column'].fillna(df['column'].median())

## 5. Remove Duplicates
df = df.drop_duplicates()

## 6. Standardize Formats
# Example: string formatting
df['Category'] = df['Category'].str.strip().str.lower()

## 7. Outlier Detection & Treatment
# Example: IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
filtered = df[~((df['column'] < (Q1 - 1.5 * IQR)) |(df['column'] > (Q3 + 1.5 * IQR)))]

## 8. Save Cleaned Dataset
filtered.to_csv("../data/cleaned_dataset.csv", index=False)