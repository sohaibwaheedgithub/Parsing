FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3.7-distutils \
    ca-certificates python3-dev python3-opencv git wget unzip


RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

ARG CLONE_REPO=1
RUN if [ "${CLONE_REPO}" = "4" ]; then git clone https://github.com/sohaibwaheedgithub/Parsing.git HPP_Repo; fi

WORKDIR /home/appuser/HPP_Repo

RUN pip install -r requirement.txt

RUN gdown 1BFVXgeln-bek8TCbRjN6utPAgRE0LJZg && mkdir checkpoint \
    && unzip JPPNet-s2.zip -d checkpoint && rm JPPNet-s2.zip

CMD python3 mod_evaluate_parsing_JPPNet-s2.py