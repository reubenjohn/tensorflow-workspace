FROM tensorflow/tensorflow:latest-gpu-py3

ARG ssh_pub_key

RUN apt-get update && apt-get install -y openssh-server

RUN mkdir /var/run/sshd

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh

# Add the keys and set permissions
RUN echo "$ssh_pub_key" > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

VOLUME ["/root/.keras/datasets", "/root/datasets/"]

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]