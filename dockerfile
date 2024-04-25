FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME \
    # 
    # change login shell to bash
    --shell /bin/bash \
    #
    # [Optional] Add sudo support.
    && apt-get update \
    && apt-get install -y sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip && \
    pip install torchinfo && \
    pip install tensorboard && \
    pip install matplotlib && \
    pip install requests && \
    rm -rf ~/.cache/pip

# [Optional] Set the default user.
USER $USERNAME
WORKDIR /work
CMD ["/bin/bash"]