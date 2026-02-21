# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)  # ← ADD THIS LINE

api = HfApi(token=HF_TOKEN)

DATASET_PATH = "hf://datasets/poojag007/superkart-sale-prediction/SuperKart.csv"
# ---- LOAD DATA ----
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['Product_Id', 'Store_Id'], errors='ignore', inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
df['Product_Sugar_Content'] = label_encoder.fit_transform(df['Product_Sugar_Content'])
df['Product_Type'] = label_encoder.fit_transform(df['Product_Type'])
df['Store_Size'] = label_encoder.fit_transform(df['Store_Size'])
df['Store_Location_City_Type'] = label_encoder.fit_transform(df['Store_Location_City_Type'])
df['Store_Type'] = label_encoder.fit_transform(df['Store_Type'])

# Define target variable
target_col = 'Product_Store_Sales_Total'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split the dataset into training and test sets
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="poojag007/superkart-sale-prediction",
        repo_type="dataset",
    )
