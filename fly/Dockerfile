FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 as base

SHELL ["/bin/bash", "-c"]

# Set path here instead of .bashrc
ENV PATH="/root/.local/bin:$PATH"

# Install core stuff
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gnupg && \
    apt-get install -y software-properties-common wget curl tmux htop sudo git neovim build-essential openssh-server python3-dev

# Configure SSH
RUN cp /etc/ssh/sshd_config /etc/ssh/sshd_config-original \
 && sed -i 's/^#\s*Port.*/Port 2222/' /etc/ssh/sshd_config \
 && sed -i 's/^#\s*PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && mkdir -p /root/.ssh \
 && chmod 700 /root/.ssh \
 && mkdir /var/run/sshd \
 && chmod 755 /var/run/sshd \
 && rm -rf /var/lib/apt/lists /var/cache/apt/archives

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["/bin/bash", "-c", "/usr/sbin/sshd && sleep inf"]
