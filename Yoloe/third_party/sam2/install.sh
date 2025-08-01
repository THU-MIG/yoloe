set -e
set -x

conda create -n sam2 python==3.10.16

conda activate sam2

pip install -r requirements.txt
pip install -e .