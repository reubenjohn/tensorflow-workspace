Custom docker image
===
Build a custom docker image for use in tensorflow_workspace experiments.  
**NOTE**: *This custom image is optional, and is not currently required to run these experiments.*

Building the image
---
The below command shows an example docker build command to build the image.
It assumes that the RSA public key is located at `~/.ssh/id_rsa.pub` and that the name of the built image is `tensorflow_workspace:latest-gpu-py3`.

    docker build --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" -t tensorflow_workspace:latest-gpu-py3 .


Features
---
 - SSH support with RSA secured password-less login: Ability to log into the docker image via SSH for additional insights and debugging.
