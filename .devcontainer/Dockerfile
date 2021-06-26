FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt update &&\
    apt install -y git &&\
    conda install -y numpy pandas scipy matplotlib tqdm jupyter jupyterlab nodejs pillow scikit-learn tensorboard &&\
    pip install opencv-python efficientnet_pytorch