# Training on DataCompDR with OpenCLIP
We provide release code and a patch to
[OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip) 
for training models on DataCompDR.

## Data
Our reinforcements to DataComp are available on HuggingFace.
- [DataCompDR-12M](https://huggingface.co/datasets/apple/DataCompDR-12M)
- [DataCompDR-12M-BFloat16](https://huggingface.co/datasets/apple/DataCompDR-12M-bf16)
- [DataCompDR-1B](https://huggingface.co/datasets/apple/DataCompDR-1B)

Our data does not include the original images and captions. For DataCompDR-12M, 
there is a corresponding 
[DataComp-12M](https://huggingface.co/datasets/mlfoundations/DataComp-12M) with 
original captions. One needs to download both datasets, then run the following 
script to join them:
```bash
#!/bin/bash
DATACOMP12M_PATH="./datasets/DataComp-12M/" # Download path of DataComp-12M from HF
DATACOMPDR12M_NOIMG_PATH="./datasets/DataCompDR-12M-noimage/" # Download path of DataCompDR-12M from HF
DATACOMPDR12M_PATH="./datasets/DataCompDR-12M/"
for  i in {00000000..00001023}
do
  mkdir tmp
  tar -xf $DATACOMP12M_PATH/${i}.tar -C tmp
  tar -xf $DATACOMP12M_NOIMG_PATH/${i}.tar -C tmp
  tar -cf $DATACOMPDR12M_PATH/${i}.tar -C tmp *.*
  rm -rf tmp
done
```

The images have to be downloaded separately. See 
[hf_dataset_example.py](../hf_dataset_example.py) for an example of downloading 
a single image.

## Installing dependencies

We use OpenCLIP for training. We have made minor modifications to OpenCLIP for 
support of loading reinforcements and the training loss. To checkout the 
specific version of each library and apply our corresponding patch run the 
following commands in order:
```bash
# Clone MobileCLIP repository
git clone git@github.com:apple/ml-mobileclip.git
cd ml-mobileclip/

# Clone OpenCLIP repository, apply patch, and install
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
git checkout cf86ee7ec4658845f640858ecd34d0f15588271a
git apply ../open_clip.patch  # Support for sampling without replacement
cp ../configs/ ./ -r
cp ../dr/ ./src/training/ -r
```

## Training

We provide scripts for training on DataCompDR-12M and DataCompDR-1B.

```bash
cd open_clip/
bash configs/run_datacomp12m.sh  # Train a ViT-B/16 on DataComp-12M without DR
bash configs/run_datacompdr12m.sh  # Train a ViT-B/16 on DataComp-12M with DR
bash configs/run_datacompdr1B.sh  # Train a ViT-B/16 on DataComp-1B with DR
```
