#default exports
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PYTHONPATH:/home/wanjiz/tpu/models"

# environment
python3 -m regionclip_venv ./regionclip_venv
source ./regionclip_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt

# RegionCLIP
git clone git@github.com:microsoft/RegionCLIP.git
python -m pip install -e RegionCLIP

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git

#python3 -m regionclip ./regionclip
#activate venv
source /home/wanjiz/vild_finetune/venv/bin/activate