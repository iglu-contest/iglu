import requests
from tqdm import tqdm


# adapted from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(id, destination, data_prefix):
    URL = "https://drive.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)
        if token is not None:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination, data_prefix)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, data_prefix):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        with tqdm(desc=f'downloading task dataset into {data_prefix}') as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(CHUNK_SIZE // 1024)
