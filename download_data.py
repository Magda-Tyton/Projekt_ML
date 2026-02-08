import kagglehub
import os
import shutil

def download_project_data():

    dest_path = 'data/data_raw.csv'
    
    download_path = kagglehub.dataset_download("clkmuhammed/autoscout24-car-listings-dataset")
    source_file = os.path.join(download_path, "autoscout24_dataset_20251108.csv")
    shutil.copy2(source_file, dest_path)

if __name__ == "__main__":
    download_project_data()