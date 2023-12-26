FROM nvcr.io/nvidia/tensorrt:23.11-py3

# Setup TimeZone
ENV TEMP_WS=/opt/inno \
    WS=/workspace \
    TIMEZONE=Asia/Taipeia 

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -yq update \
    && apt-get -yq install tzdata \
    && ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Install Basic Packages
RUN apt-get install -qy \
    libgtk2.0-dev pkg-config yasm \
    ffmpeg libsm6 libxext6 libcanberra-gtk-module libcanberra-gtk3-module \
    vim python3-dev python3-pip
    
# About Python Modules
RUN python3 -m pip install pip --upgrade

# About sias
COPY [ "src/", "/opt/src/" ]
COPY [ "setup.py", "requirements.txt", "/opt/"]
WORKDIR "/opt"
RUN pip install -e .

WORKDIR /workspace
