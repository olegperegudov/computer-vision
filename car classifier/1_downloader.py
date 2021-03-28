# will dl images here and create the folder for each query
# nice to quickly download some images
# not used in the main script. It just for testing/debugging

from bing_image_downloader import downloader
import os
from pathlib import Path

dl_path = Path('data')

query_string = ['bmw 3 blue',
                'bmw x6 2020 red',
                'tesla 3 white']


for query in query_string:
    downloader.download(query,
                        limit=50,
                        output_dir=dl_path,
                        adult_filter_off=True,
                        force_replace=False,
                        timeout=60)
