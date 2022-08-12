#clone repo
git clone https://github.com/Jawing/RegionCLIP.git

#default exports
export PYTHONUNBUFFERED=1

# environment
python3.9 -m venv ./regionclip_venv
source ./regionclip_venv/bin/activate
chmod -R 777 ./regionclip_venv
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# RegionCLIP
python3 -m pip install -e RegionCLIP

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/Jawing/object_detection_metrics.git
pip install shapely
pip install flask
#python3 -m pip install -r ./RegionCLIP/requirements.txt

#put images here for inference
mkdir ./RegionCLIP/datasets/custom_images