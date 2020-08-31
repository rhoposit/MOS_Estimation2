# MOS_Estimation2
Apply a pre-trained model to wav files to obtain a MOS quality score
Based on work reported in: "Comparison of Speech Representations for Automatic Quality Estimation in Multi-Speaker Text-to-Speech Synthesis"
https://arxiv.org/abs/2002.12645
( Note this code was derived from https://github.com/lochenchou/MOSNet, https://arxiv.org/abs/1904.08352 )


# Dependency
Linux Ubuntu 16.04
- GPU: GeForce RTX 2080 Ti
- Driver version: 418.67
- CUDA version: 10.1

Python 3.5
- tensorflow-gpu==2.0.0-beta1 (cudnn=7.6.0)
- scipy
- pandas
- matplotlib
- librosa

### Environment set-up
For example,
```
conda create -n mosnet python=3.5
conda activate mosnet
pip install -r requirements.txt
conda install cudnn=7.6.0
```

# Usage
1. place all speech wavfiles into a directory, AUDIO_DIR
2. Run `python utils.py AUDIO_DIR` to prepare the wav files. 
3. Run `python apply_model.py` to generate MOS scores
4. Final results will be provided in a CSV file, in `data/results.txt`, as well as a pickle format (useful for plotting, using origial MOSnet code) `data/results.pkl`

Note a small example using VCTK data is provided, the wav files are in the directory `audio/`. Also note that this example is set up to parse speaker ID from the filename. You can change this, based on your own file name convention, as well as using different systems, by modifying line 76 in `apply_model.py`