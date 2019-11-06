# Speech-Recognition-with-3D-Waves
Just happing 3D Wavves
Version
Current Version : 0.0.0.2
#Dependencies ( VERSION MUST BE MATCHED EXACTLY!)
import os
from pathlib import Path
import IPython.display as ipd

import os
from os.path import isdir, join
from pathlib import Path

import pandas as pd 
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
from collections import Counter
%matplotlib inline

from subprocess import check_output
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

#Problem Statements in detail Explaination as per below 

According to datasets few informational files and a folder of audio files.The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that tried to predicted. The labels trying to predict in Test are yes, no, up, down, left, right, on, off, stop, go. Everything else should be considered either unknown or silence. The folder _background_noise_ contains longer clips of "silence" we used break up and use as training input.
The files contained in the training audio are not uniquely named across labels, For example, 00f0204f_nohash_0.wav is found in 14 folders, but that file is a different speech command in each folder. Finally we got 00f0204f_nohash_0.wav
Twenty core command words were recorded, with most speakers saying each of them five times. The core words are "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine". To help distinguish unrecognized words, there are also ten auxiliary words, which most speakers only said once. These include "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", and "Wow".


#Datasets

*sample_submission.csv - A sample submission file in the correct format.
This is a set of one-second .wav audio files, each containing a single spoken English word. These words are from a small set of commands, and are spoken by a variety of different speakers. The audio files are organized into folders based on the word they contain, and this data set is designed to help train simple machine learning models.

#Datasets info-This is version 0.01 of the data set containing 64,727 audio files

happy/3cfc6b3a_nohash_2.wav indicates that the word spoken was "happy", the speaker's id was "3cfc6b3a", and this is the third utterance of that word by this speaker in the data set. The 'nohash' section is to ensure that all the utterances by a single speaker are sorted into the same training partition, to keep very similar repetitions from giving unrealistically optimistic evaluation scores.

# Partitioning
The audio clips haven't been separated into training, test, and validation sets explicitly, but by convention a hashing function is used to stably assign each file to a set. Here's some Python code demonstrating how a complete file path and the desired validation and test set sizes (usually both 10%) are used to assign a set:

```python MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 # ~134M

def which_set(filename, validation_percentage, testing_percentage): """Determines which data partition the file should belong to.

We want to keep files in the same training, validation, or testing sets even if new ones are added over time. This makes it less likely that testing samples will accidentally be reused in training when long runs are restarted for example. To keep this stability, a hash of the filename is taken and used to determine which set it should belong to. This determination only depends on the name and the set proportions, so it won't change as other files are added.

It's also useful to associate particular files as related (for example words spoken by the same person), so anything after 'nohash' in a filename is ignored for set determination. This ensures that 'bobby_nohash_0.wav' and 'bobby_nohash_1.wav' are always in the same set, for example.

Args: filename: File path of the data sample. validation_percentage: How much of the data set to use for validation. testing_percentage: How much of the data set to use for testing.

Returns: String, one of 'training', 'validation', or 'testing'. """ base_name = os.path.basename(filename) # We want to ignore anything after 'nohash' in the file name when # deciding which set to put a wav in, so the data set creator has a way of # grouping wavs that are close variations of each other. hash_name = re.sub(r'nohash.*$', '', base_name) # This looks a bit magical, but we need to decide whether this file should # go into the training, testing, or validation sets, and we want to keep # existing files in the same set even if more files are subsequently # added. # To do that, we need a stable way of deciding based on just the file name # itself, so we do a hash of that and then use that to generate a # probability value that we use to assign it. hash_name_hashed = hashlib.sha1(hash_name).hexdigest() percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS)) if percentage_hash < validation_percentage: result = 'validation' elif percentage_hash < (testing_percentage + validation_percentage): result = 'testing' else: result = 'training' return result ```

The results of running this over the current set are included in this archive as validation_list.txt and testing_list.txt. These text files contain the paths to all the files in each set, with each path on a new line. Any files that aren't in either of these lists can be considered to be part of the training set.



