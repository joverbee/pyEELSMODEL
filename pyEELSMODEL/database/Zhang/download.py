import requests
import os 

url = 'https://zenodo.org/records/11199911/files/Dirac_GOS.gosh'
dir_path = os.path.dirname(os.path.dirname(__file__) + "/../pyEELSMODEL/database/Zhang/")
filename = dir_path + 'Dirac_GOS.gosh'

def download_file(url = url, filename = filename):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file with write-binary mode
        with open(filename, 'wb') as f:
            # Write the content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded '{filename}' successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")