# Base OS
FROM alpineintuition/base:latest

MAINTAINER Tanmay Thakur <tanmaythakurbrn2rule@gmail.com>

# Install Build Utilities
RUN apt-get update && \
	#apt-get install -y gcc make apt-transport-https ca-certificates build-essential && \
	apt-get install -y libsndfile-dev



RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip 


RUN apt install -y  python-pip && \
    pip install -U pip



# Check Python Environment
RUN python --version
RUN pip --version

# set the working directory for containers
WORKDIR .



# Test Env
#RUN ./test.sh

ARG UID
ARG GID
# Create user
RUN groupadd --gid $GID docker
RUN useradd --uid $UID --gid docker --shell /bin/zsh --create-home user
WORKDIR /home/user

# Copy Files
COPY .  /home/user/Voice-synthesis
WORKDIR /home/user/Voice-synthesis


# Install Dependencies
RUN ./run.sh
WORKDIR /home/user
RUN rm -rf Voice-synthesis
WORKDIR /home/user/dev

RUN chown -R user:docker /home/user
USER user

# Running the server
#CMD ["python", "app.py"] 
