import requests
import zipfile
import os

path_name = "."

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url} (status code: {response.status_code})")

# Example: Download TinyImageNet
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
save_path =  path_name + "/tiny-imagenet-200.zip"
download_file(url, save_path)

# unzip the dataset
def unzip_file(zip_path, extract_to):
    """
    Unzips a file to the specified directory.
    
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Example: Unzipping TinyImageNet
zip_path = path_name + "/tiny-imagenet-200.zip"  # Path to the zip file
extract_to = path_name + "/tiny-imagenet-200"  # Directory to extract to

# Create the directory if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Unzip the file
unzip_file(zip_path, extract_to)
