# Download our demo project:
import urllib.request
from io import BytesIO
from zipfile import ZipFile

def unzip_from_url(url, dest_folder=''):
    # Directly extract files without writing the archive to disk
    resp = urllib.request.urlopen(url)
    with ZipFile(BytesIO(resp.read())) as zf:
        zf.extractall(path=dest_folder)


project_url = "http://deeplabcut.rowland.harvard.edu/datasets/demo-me-2021-07-14.zip"
unzip_from_url(project_url, "/content")