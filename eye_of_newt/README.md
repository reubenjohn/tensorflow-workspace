Eye of Newt
===
![Early](https://img.shields.io/badge/Status-Very%20Early%20Stage-yellow)

Model the rich visual dynamics of diverse video sequences using unsupervised techniques to act as a powerful prior to be transferred to further experiments.  

### Roadmap
#### Phase 1. [Reproduce input image sanity check](experiments/image_reconstruction_sanity.py) 
![Status|Complete](https://img.shields.io/badge/Status-Complete-brightgreen)
   
Ability to reproduce one of three input images from the training data using a small scale auto-encoder.
This experiment is a sanity check and establishes a basic implementation style.

|![Reconstruction Sanity](screenshots/reconstruction-sanity-outputs.png)|![Reconstruction Sanity Graph](screenshots/reconstruction-sanity-model.png)|
|---|---|
|Screenshots of input images and corresponding reconstructions at step 210(left), 300(middle) and 5640(right).|Simple model with essentially only 2 dense layers|

#### Phase 2. Generalized image reproduction  
![Early](https://img.shields.io/badge/Status-In%20Progress-yellow)

Ability to reproduce input images outside the training data.

#### Phase 3. Next frame prediction sanity check  
![Early](https://img.shields.io/badge/Status-Pending-grey)

Ability to predict the next frame of a video sequence from the training data.

#### Phase 4. Generalized next frame prediction  
![Early](https://img.shields.io/badge/Status-Pending-grey)
  
Ability to predict the next frame of a video sequence outside the training data.
