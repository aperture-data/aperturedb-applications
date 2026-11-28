# Using HMDB51 to train movinet.

[Download it here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

[Pytorch implementation](https://pytorch.org/vision/main/generated/torchvision.datasets.HMDB51.html)

### Needed packages and binaries.

#### Mac
    brew brew install rar ffmpeg
#### Ubuntu
    apt install rar on ubuntu.

### Download data
    mkdir input splits
    DOWNLOAD=true bash prepare_data.sh

### Download the helper scripts and classes.
    wget https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/AHMDB51.py
    wget https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/AVideoClips.py
    wget https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/ingest_transcode.py
    wget https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/train_movinet.py
    wget https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/prepare_data.sh


### Ingest into aperturedb
    python ingest_transcode.py

### Train model
    python train_movinet.py training true

### Explore classification abilities with the "off the shelf moveinet" vs fine tuned version of the same.
[Classify Vanilla trained](https://github.com/aperture-data/aperturedb-applications/blob/train_movinet/training/movinet_with_hmdb51/Classify-Vanilla-trained.ipynb)

## Glossary of the files and resources.
### AHMDB51.py
It is an subclass of HMDB51 (from pytorch) and it incorporates the fact that the videos are stored in ApertureDB rather than as local files.

### AVideoClips.py
Video Clips resample the Videos into clips of 16 frame lengths, sampled at a specified fps.
Since Movinet expects inputs of 172x172, or 200x200 pixels, there's also a transformation that is applied to a batch of videos.
