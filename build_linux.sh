# conda update -n base conda	# Upgrade conda
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge	# Download ffmpeg and x264encoder
python -m pip install --upgrade pip
pip3 install -r requirements.txt
# conda install --yes --file requirements.txt

