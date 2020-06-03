from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
#import argparse
import torch
#import sys

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from tqdm import tqdm


if __name__ == '__main__':
    synth_path = Path("../src/pre_train/synthesizer/saved_models/logs-pretrained/")
    vocoder_path = Path("../src/pre_train/vocoder/saved_models/pretrained/pretrained.pt")
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    

    ## Load the models one by one.
    print("Preparing the synthesizer and the vocoder...")
    synthesizer = Synthesizer(synth_path.joinpath("taco_pretrained"), low_mem=False)
    vocoder.load_model(vocoder_path)
    
    print("Loading encoder from resemblyzer")
    encoder = VoiceEncoder()
    
    # Get the reference audio repo path
    speaker = 'SAM'
    repo_fpath = Path('../SOURCE_AUDIO',speaker)
    wav_fpaths = list(repo_fpath.glob(speaker+"*"))
    print('PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATHS')
    print(repo_fpath)
    print(wav_fpaths)
    
    wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths)))))
    speaker_embedding = encoder.embed_speaker(wavs)
    
    text = str(np.loadtxt('../test_sentence.txt', dtype='str', delimiter = '&'))
   
    texts = [text]
    embeds = [speaker_embedding]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    
    
    generated_wav = vocoder.infer_waveform(spec)
    
    
    # Save it on the disk
    fpath = "outputs/test_output_"+ speaker + ".wav"
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % fpath)
