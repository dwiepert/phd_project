# PhD project
Conferences to consider:
* CARE Research Day: April 17, 2026
* 2026 ASHA: November 19-21, 2026, Indiana 
    * Submission deadline: TODO
* 2026 IEEE Spoken Language Technology Workshop (SLT): December 13-16, 2026, Palermo, Italy
    * Submission deadline: TODO

# Package install
`conda install ffmpeg`
`cd  ./phd_project`
`pip install .`

```
git clone https://github.com/cheoljun95/Speech-Articulatory-Coding.git
cd Speech-Articulatory-Coding
pip insall -e .
```

# Data format 
Expectation: each audio file has a unique file name - this is the unique identified (uid) for UidToWaveform. 
There exists a metadata file linking uid to task (for filtering metadata) and speakerID (for splitting). 
SEE helper functions.
Figure out unexpandable/unreadable files
 
# Model Pipeline
![Training](figures/training.png)

# TODO
1. Add way for date col to HAVE to be convertable to datetime format??
3. SPARC model implementation - separate EMA and resynth, separate speaker identity
4. WER/CER 
5. UTMOS
6. Speaker/vector similarity metrics
7. Start building conversion model
8. Split data
9. Account for difference between UID and file name column

Defaults to speaker split if both splits are toggled on

# Mayo Meeting
The model needs to be UT IP. Only thing I'm not sure about is who on the mayo side can help draft an agreement like that. 

IDEA:
make a transform that transforms into EMA and saves the output somewhere if specified OR loads EMA if it already exists
make a test to see how long it takes to transform into EMA without a gpu/if that is even possible