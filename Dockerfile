FROM ubuntu
MAINTAINER A-Wels 
# make /bin/sh symlink to bash instead of dash:
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash
# set ENV to execute startup scripts
ENV ENV ~/.profile
ENV CONDA_DIR /opt/conda

RUN apt-get update && \
	apt-get install -y build-essential && \
	apt-get install -y wget unzip && \
	apt-get install -y git && \
	apt-get clean



ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda && \
	conda init bash
#	source ~/.bashrc  && \
RUN python -V
RUN conda install -c anaconda -y python=3.7.15
RUN	conda install -y pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch  && \
	pip install yacs loguru einops timm==0.4.12 imageio
RUN cd ~ &&\
	git clone https://github.com/A-Wels/FlowFormer-Official && \
	cd FlowFormer-Official && \
	mkdir checkpoints && \
	cd checkpoints  && \
	wget https://cloud.a-wels.de/index.php/s/zwpYsbG7jAFHfLQ/download/FlowFormer-Models-20221231T113622Z-001.zip  && \
	unzip FlowFormer-Models-20221231T113622Z-001.zip && \
	mv FlowFormer-Models/* . &&\
	rm -r FlowFormer-Models && \
	rm FlowFormer-Models-20221231T113622Z-001.zip  && \
	cd ..  && \
	mkdir demo_data

ENTRYPOINT ["/bin/bash"]
