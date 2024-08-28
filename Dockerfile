FROM python:3.8

ENV LANG C.UTF-8

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y apt-utils lsof tree vim \
   net-tools htop curl wget ssh git git-lfs

RUN python --version
RUN pip install Cython urllib3 psutil certifi
RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu

COPY . /aiapp1
WORKDIR /aiapp1

RUN pip install -e .[api]

EXPOSE 8000

CMD ["uvicorn", "aiapp1.app:app", "--host=0.0.0.0", "--port=8000"]
