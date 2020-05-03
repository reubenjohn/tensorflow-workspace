Eye of Newt
===
![Early](https://img.shields.io/badge/Status-Very%20Early%20Stage-yellow)

Model the rich visual dynamics of diverse video sequences using unsupervised techniques to act as a powerful prior to be transferred to further experiments.  

### Roadmap

---

#### Phase 1. [Reproduce input image sanity check](experiments/image_reconstruction_sanity.py) 
![Status|Complete](https://img.shields.io/badge/Status-Complete-brightgreen)
   
Ability to reproduce one of three input images from the training data using a small scale auto-encoder.
This experiment is a sanity check and establishes a basic implementation style.

|![Reconstruction Sanity](screenshots/reconstruction-sanity-outputs.png)|![Reconstruction Sanity Graph](screenshots/reconstruction-sanity-model.png)|
|---|---|
|Screenshots of input images and corresponding reconstructions at step 210(left), 300(middle) and 5640(right).|Simple model with essentially only 2 dense layers|
|**Training time**|~2min to ~0 loss|

---

#### Phase 2. Deep image reproduction
![Early](https://img.shields.io/badge/Status-Final%20Stage-lightgreen)

Same as Phase 1, but with deep neural network architectures.

|Architecture|Model|Output|
|---|---|---|
|**V1: Shallow Conv-Deconv**
1-Conv-1-Deconv|![Shallow Model][1]|![Shallow Output][2]|
|**V2: Deep Conv-Deconv**
4-Conv-4-Deconv Cannot converge to 100% accuracy due to loss of fine grained spacial information in middle layers|![Deep Model][3]|![Deep Output][4]|
|**V3: Deep With Skip Connections**
Converges faster to 97% accuracy as compared to v2 by introducing skip connections from input to deep layers. However, obviously, non-skip connections eventually die as network learns to exploit skip connections!|![Deep Skip Model][5]|![Deep Output][6]|
|**V4: Deep With Noisy Skip Connections (WIP)**
Noise in skip connection forces network to learn high-dimensional embeddings of input, whilst leveraging fine grained details of input. Great stepping-stone to phase 4 where inputs don't equal targets|-|-|

[1]: screenshots/reconstruction-image-sanity-shallow-conv-model.png
[2]: screenshots/reconstruction-image-sanity-shallow-conv-output.png
[3]: screenshots/reconstruction-image-sanity-deep-model.png
[4]: screenshots/reconstruction-image-sanity-deep-output.png
[5]: screenshots/reconstruction-image-sanity-deep-skip-model.png
[6]: screenshots/reconstruction-image-sanity-deep-skip-output.png

##### Training
![Training Loss and MSE](screenshots/phase2-training.png)

|Pink|Green|Gray|Training time|
|---|---|---|---|
|Shallow Architecture|Deep Architecture|Deep Skip Connection|~45min|

---

#### Phase 3. Generalized image reproduction  
![Early](https://img.shields.io/badge/Status-Pending-grey)

Ability to reproduce input images outside the training data.

---

#### Phase 4. Next frame prediction sanity check  
![Early](https://img.shields.io/badge/Status-Pending-grey)

Ability to predict the next frame of a video sequence from the training data.

---

#### Phase 5. Generalized next frame prediction  
![Early](https://img.shields.io/badge/Status-Pending-grey)
  
Ability to predict the next frame of a video sequence outside the training data.
