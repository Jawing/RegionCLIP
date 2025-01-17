#clone repo
#git clone https://gitlab.com/humanware.ca/platforms/ai/RegionCLIP.git

#default exports
export PYTHONUNBUFFERED=1

# setup environment
sudo rm -r /data/venv/regionclip
sudo python3.9 -m venv /data/venv/regionclip
source /data/venv/regionclip/bin/activate
sudo chmod -R 777 /data/venv/regionclip
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# RegionCLIP
python3 -m pip install -e RegionCLIP

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://gitlab.com/humanware.ca/platforms/ai/object_detection_metrics.git
pip install shapely==1.8.2
pip install bbox-visualizer
#python3 -m pip install -r ./RegionCLIP/requirements.txt

pip install albumentations==1.2.1
#mlflow setup for training experiments
pip install mlflow
# Set the experiment via environment variables
export MLFLOW_EXPERIMENT_NAME=RegionCLIP-whereismystuff
mlflow experiments create --experiment-name RegionCLIP-whereismystuff

#dash and jupyter-dash visualization tools
pip install --q dash==2.0.0 jupyter-dash==0.4.0;
pip install dash-bootstrap-components

#put images here for inference
mkdir ./RegionCLIP/datasets/custom_images