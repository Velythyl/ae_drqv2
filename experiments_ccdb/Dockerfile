FROM python:3.9.12-bullseye
RUN apt-get update && apt-get upgrade -y
RUN apt install git -y

RUN pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
ENTRYPOINT python3 train.py