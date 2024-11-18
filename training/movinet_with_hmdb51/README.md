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

### Ingest into aperturedb
    python ingest_transcode.py

