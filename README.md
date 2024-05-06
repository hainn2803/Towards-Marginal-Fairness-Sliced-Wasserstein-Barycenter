# No idea

## Environment Installation

This repo requires `Python 3.10.12`.
Run this command: `conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
Then run this command: `pip install -r requirements.txt`

## Training script:
Run this command: `bash train_run.bash`

## Evaluating script:
After training, we evaluate all methods 10 times and conduct average value for each metric by running this script: `bash evaluate.bash`