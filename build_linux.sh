#! /bin/sh

echo "Start building..."
# Download ffmpeg and x264encoder
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge 
pip3 install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:./Model/"
echo "Building ends..."

