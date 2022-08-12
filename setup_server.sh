#clone repo
#git clone https://github.com/Jawing/RegionCLIP.git
#git clone --single-branch --branch server https://github.com/Jawing/RegionCLIP.git
#default exports
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# environment
conda create -n regionclip python=3.9
source activate regionclip
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# RegionCLIP
python3 -m pip install -e RegionCLIP

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/Jawing/object_detection_metrics.git
pip install shapely
pip install flask
#python3 -m pip install -r ./RegionCLIP/requirements.txt

#create missing directories
mkdir ./RegionCLIP/datasets/custom_images
mkdir ./RegionCLIP/models
mkdir -p ./RegionCLIP/output/concept_feats
mkdir -p ./RegionCLIP/pretrained_ckpt/rpn

#import model/concept weights

#run server (start within regionclip dir)
cd ./RegionCLIP
python server.py