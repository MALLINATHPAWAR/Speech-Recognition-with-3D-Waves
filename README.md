# Speech-Recognition-with-3D-Waves
Just happing 3D Wavves
Version
Current Version : 0.0.0.2
Dependencies ( VERSION MUST BE MATCHED EXACTLY!)
tensorflow == 1.0.0
sugartensor == 1.0.0.2
pandas >= 0.19.2
librosa == 0.5.0
scikits.audiolab==0.11.0
If you have problems with the librosa library, try to install ffmpeg by the following command. ( Ubuntu 14.04 )


sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get -y install ffmpeg
Dataset
We used VCTK, LibriSpeech and TEDLIUM release 2 corpus. Total number of sentences in the training set composed of the above three corpus is 240,612. Valid and test set is built using only LibriSpeech and TEDLIUM corpuse, because VCTK corpus does not have valid and test set. After downloading the each corpus, extract them in the 'asset/data/VCTK-Corpus', 'asset/data/LibriSpeech' and 'asset/data/TEDLIUM_release2' directories.

Audio was augmented by the scheme in the Tom Ko et al's paper. (Thanks @migvel for your kind information)

Pre-processing dataset
The TEDLIUM release 2 dataset provides audio data in the SPH format, so we should convert them to some format librosa library can handle. Run the following command in the 'asset/data' directory convert SPH to wave format.


find -type f -name '*.sph' | awk '{printf "sox -t sph %s -b 16 -t wav %s\n", $0, $0".wav" }' | bash
If you don't have installed sox, please installed it first.


sudo apt-get install sox
We found the main bottle neck is disk read time when training, so we decide to pre-process the whole audio data into the MFCC feature files which is much smaller. And we highly recommend using SSD instead of hard drive.
Run the following command in the console to pre-process whole dataset.


python preprocess.py
Training the network
Execute


python train.py ( <== Use all available GPUs )
or
CUDA_VISIBLE_DEVICES=0,1 python train.py ( <== Use only GPU 0, 1 )
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory. Launch tensorboard --logdir asset/train/log to monitor training process.

We've trained this model on a 3 Nvidia 1080 Pascal GPUs during 40 hours until 50 epochs and we picked the epoch when the validatation loss is minimum. In our case, it is epoch 40. If you face the out of memory error, reduce batch_size in the train.py file from 16 to 4.

The CTC losses at each epoch are as following table:
