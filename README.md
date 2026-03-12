# LuxTTS
<p align="center">
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Colab Notebook">
  </a>
</p>

LuxTTS is an lightweight zipvoice based text-to-speech model designed for high quality voice cloning and realistic generation at speeds exceeding 150x realtime.

https://github.com/user-attachments/assets/a3b57152-8d97-43ce-bd99-26dc9a145c29


### The main features are
- Voice cloning: SOTA voice cloning on par with models 10x larger.
- Clarity: Clear 48khz speech generation unlike most TTS models which are limited to 24khz.
- Speed: Reaches speeds of 150x realtime on a single GPU and faster then realtime on CPU's as well.
- Efficiency: Fits within 1gb vram meaning it can fit in any local gpu.
- **NEW**: Gradio GUI with auto-duration detection and manual reference text support
- **NEW**: ROCm support for AMD GPUs

## Usage
You can try it locally, colab, or spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YatharthS/LuxTTS)

#### Simple installation:
```bash
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt
```

#### ROCm Installation (AMD GPUs):
For AMD GPU users with ROCm PyTorch builds, you'll need to install necessary rocm and pytorch packages.

### Quick Start with GUI

Launch the Gradio interface for an easy-to-use web UI:

```bash
python gui.py
```

Features:
- **Auto-duration detection**: Automatically matches duration to your audio length
- **Manual reference text**: Optional manual transcription to skip Whisper (faster)
- **Auto-transcribe button**: Preview and edit Whisper's transcription
- **All parameters**: Full control over RMS, duration, steps, guidance scale, speed, and smoothing

### Load model:
```python
from zipvoice.luxvoice import LuxTTS

# load model on GPU
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# load model on CPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

# load model on MPS for macs
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')

# Skip Whisper loading (if providing manual reference text)
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda', skip_whisper=True)
```

### Simple inference
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```

### Inference with all parameters:
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## Encoding parameters
rms = 0.01          # Volume normalization (0.01 recommended)
duration = 10       # Reference audio duration in seconds

## Generation parameters
num_steps = 4       # Sampling steps (3-4 best for efficiency, higher = better quality)
guidance_scale = 3.0 # CFG strength (higher = more faithful to prompt)
t_shift = 0.5       # Sampling temperature (default: 0.5, higher may improve quality)
speed = 1.0         # Playback speed (lower = slower)
return_smooth = False # Smoother output but less clean

## encode audio (takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=duration, rms=rms)

## generate speech
final_wav = lux_tts.generate_speech(
    text, 
    encoded_prompt, 
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    t_shift=t_shift,
    speed=speed,
    return_smooth=return_smooth
)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```

### Advanced: Manual Reference Text (Skip Whisper)

Provide manual transcription to skip Whisper processing and save time:

```python
## Manual reference text (must match audio exactly)
reference_text = "This is exactly what is said in my reference audio"

## encode with manual text
encoded_prompt = lux_tts.encode_prompt(
    prompt_audio, 
    duration=duration, 
    rms=rms,
    reference_text=reference_text  # Skip Whisper transcription
)
```

**Important**: The reference text must match **exactly** what's in the first `duration` seconds of your audio. For best results, use the auto-transcribe feature in the GUI to see what Whisper detects.

## Tips
- **Duration matching is critical**: Set `duration` to match your actual audio length for best results
  - Example: 8-second audio → `duration=8`
  - The GUI automatically detects and sets this for you
- Use at minimum a 3 second audio file for voice cloning
- You can use `return_smooth = True` if you hear metallic sounds
- Lower t_shift for less possible pronunciation errors but worse quality and vice versa
- For ROCm users: The `k2` warning is normal and can be ignored (falls back to PyTorch implementation)

## What's New in This Fork

### Gradio GUI (`demo.py`)
- **Auto-duration detection**: Automatically sets duration to match your audio file
- **Manual reference text support**: Optional field to provide transcription manually
- **Auto-transcribe button**: Preview Whisper's output before generation
- **All parameters exposed**: RMS, duration, steps, guidance scale, t_shift, speed, smoothing
- **Clean, intuitive interface**: Organized into Basic and Advanced parameters

### Manual Reference Text Support
- Added `reference_text` parameter to `encode_prompt()`
- Added `skip_whisper` option to skip Whisper model loading entirely
- Saves 2-5 seconds per generation when using manual text
- Saves ~500MB+ memory when Whisper is not loaded

### ROCm Compatibility
- Patch for `torch.distributed.ReduceOp` missing in ROCm builds
- Works on AMD GPUs with ROCm PyTorch
- No distributed training support needed for inference

## File Structure

```
LuxTTS/
├── demo.py                      # Gradio GUI (recommended)
├── patch_torch_distributed.py   # ROCm compatibility patch
├── patches/
│   └── distrib.py              # Fixed encodec distrib.py for ROCm
├── zipvoice/
│   ├── luxvoice.py             # Updated with skip_whisper and reference_text
│   └── modeling_utils.py       # Updated with reference_text support
└── requirements.txt
```

## Info

Q: How is this different from ZipVoice?

A: LuxTTS uses the same architecture but distilled to 4 steps with an improved sampling technique. It also uses a custom 48khz vocoder instead of the default 24khz version.

Q: Can it be even faster?

A: Yes, currently it uses float32. Float16 should be significantly faster(almost 2x).

Q: Does it work on AMD GPUs?

A: Yes! This fork includes ROCm support. See the ROCm installation section above.

Q: Why does duration need to match audio length?

A: The model aligns audio features with text tokens. If duration doesn't match the actual audio length, you get a mismatch between what the audio contains and what the text describes, resulting in garbled output.

## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [x] Release MPS support (thanks to @builtbybasit)
- [x] **Gradio GUI with auto-duration detection**
- [x] **Manual reference text support**
- [x] **ROCm/AMD GPU support**
- [ ] Release LuxTTS v1.5
- [ ] Release code for float16 inference

## Acknowledgments

- [ZipVoice](https://github.com/k2-fsa/ZipVoice) for their excellent code and model.
- [Vocos](https://github.com/gemelo-ai/vocos.git) for their great vocoder.
- Original LuxTTS by [YatharthS](https://github.com/ysharma3501/LuxTTS)
  
## Final Notes

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Original author: yatharthsharma350@gmail.com
