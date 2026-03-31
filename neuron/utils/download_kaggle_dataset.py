import os

# SET YOUR CREDENTIALS HERE
os.environ['KAGGLE_USERNAME'] = "francklayman@gmail.com"
os.environ['KAGGLE_KEY'] = "KGAT_9676607a324b23174c120da3ba77b3fb"

# Now import the API
from kaggle.api.kaggle_api_extended import KaggleApi

def download_houston_data():
    api = KaggleApi()
    api.authenticate()
    
    dataset = 'ahmedshahriarsakib/usa-real-estate-dataset'
    print(f"Downloading {dataset}...")
    
    # Download and unzip in current directory
    api.dataset_download_files(dataset, path='.', unzip=True)
    print("Done. realtor-data.csv is ready.")

if __name__ == "__main__":
    download_houston_data()