conda remove -n mapfree --all
conda create -n mapfree python=3.9
conda activate mapfree
module load cuda/11.8
conda install tobiasrobotics::open3d -c conda-forge
conda install pytorch==2.0.1 torchvision pytorch-cuda==11.8 -c pytorch -c nvidia -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c conda-forge python_abi
conda install pytorch3d -c pytorch3d -c conda-forge
pip install opencv-python
# pip install open3d=0.15.1
pip install transforms3d
# pip install yacs
pip install h5py
pip install pytorch-lightning==1.6.5
pip install kornia==0.6.11
pip install hydra-core
pip install omegaconf
pip install einops
pip install visdom

# pip install open3d
# pip3 install open3d-python -U
# conda install open3d-admin::open3d
# pip3 install open3d
# pip uninstall open3d-python 
# pip3 install open3d
# conda install saedrna::open3d
# conda install conda-forge::yacs

git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc
git clone --recursive https://github.com/naver/dust3r.git dependency/dust3r
cd dependency/hloc
python -m pip install -e .
cd ../../