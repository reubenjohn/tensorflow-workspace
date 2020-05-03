FROM tensorflow/tensorflow:latest-gpu-py3

ARG ssh_pub_key

# Install Dependencies

RUN apt-get update

RUN apt-get install -y graphviz

RUN pip install tensorflow_datasets pydot pydotplus graphviz

# SSH Setup

RUN apt-get install -y openssh-server graphviz

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