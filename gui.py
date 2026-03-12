import gradio as gr
import numpy as np
import soundfile as sf
import tempfile
import os
from zipvoice.luxvoice import LuxTTS

# Initialize LuxTTS model (only once at startup)
print("Loading LuxTTS model...")
# Set skip_whisper=True to skip Whisper loading if you'll provide reference text manually
# This saves memory and loading time
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda', threads=2, skip_whisper=False)  # Use device='cpu' for CPU
print("Model loading complete")

def get_audio_duration(audio):
    """Get the actual duration of the uploaded audio"""
    if audio is None:
        return None
    
    try:
        sample_rate, audio_data = audio
        duration_seconds = len(audio_data) / sample_rate
        return round(duration_seconds, 2)
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None


def transcribe_reference(audio, duration, rms):
    """Helper function to transcribe reference audio and populate the text field"""
    if audio is None:
        return "❌ Please upload an audio file first"
    
    try:
        import librosa
        from zipvoice.utils.infer import rms_norm
        
        sample_rate, audio_data = audio
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sample_rate)
        
        # Load and transcribe just like process_audio does
        print(f"Transcribing {duration} seconds of reference audio...")
        prompt_wav2, sr = librosa.load(tmp_path, sr=16000, duration=duration)
        
        # Use the transcriber directly
        if lux_tts.transcriber is None:
            os.unlink(tmp_path)
            return "❌ Whisper not available. Initialize with skip_whisper=False"
        
        whisper_output = lux_tts.transcriber(prompt_wav2)
        transcribed_text = whisper_output["text"].strip()
        
        # Delete temporary file
        os.unlink(tmp_path)
        
        print(f"✓ Transcribed: '{transcribed_text}'")
        return transcribed_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def generate(audio, text, reference_text, rms, duration, num_steps, t_shift, guidance_scale, speed, return_smooth):
    if audio is None:
        return None, "Error: Please upload an audio file"
    
    if not text or text.strip() == "":
        return None, "Error: Please enter text"
    
    try:
        sample_rate, audio_data = audio
        
        print(f"Received text: {text}")
        print(f"Reference text: {reference_text if reference_text else 'Will auto-transcribe'}")
        print(f"Sample rate: {sample_rate}")
        print(f"Audio data shape: {audio_data.shape}")
        print(f"Parameters - RMS: {rms}, Duration: {duration}, Num steps: {num_steps}, T-shift: {t_shift}, Guidance scale: {guidance_scale}, Speed: {speed}, Return smooth: {return_smooth}")
        
        # Save audio to temporary file (LuxTTS requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sample_rate)
        
        # Encode audio (first run may take ~10 seconds)
        print("Encoding audio...")
        # Use manual reference text if provided, otherwise auto-transcribe with Whisper
        ref_text = reference_text.strip() if reference_text and reference_text.strip() else None
        encoded_prompt = lux_tts.encode_prompt(tmp_path, duration=duration, rms=rms, reference_text=ref_text)
        
        # Generate speech
        print("Generating speech...")
        final_wav = lux_tts.generate_speech(
            text, 
            encoded_prompt, 
            num_steps=num_steps, 
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            speed=speed,
            return_smooth=return_smooth
        )
        
        # Convert to numpy array
        final_wav = final_wav.numpy().squeeze()
        
        # Delete temporary file
        os.unlink(tmp_path)
        
        print("Speech generation complete")
        
        # Return with 48000Hz sample rate
        return (48000, final_wav), "✓ Speech generation complete"
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LuxTTS Voice Cloning")
    gr.Markdown("Upload a reference audio and enter text to generate speech")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Reference Audio (WAV/MP3)",
                type="numpy",
                sources=["upload", "microphone"]
            )
            
            reference_text_input = gr.Textbox(
                label="Reference Text (Optional)",
                placeholder="Enter what is said in the reference audio (leave empty to auto-transcribe with Whisper)",
                lines=2
            )
            
            with gr.Row():
                transcribe_btn = gr.Button("🎤 Auto-Transcribe Reference Audio", size="sm")
                
            gr.Markdown("*Tip: Click Auto-Transcribe to see what Whisper detects, then edit if needed*")
            
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to generate as speech",
                lines=3
            )
            
            # Parameter controls
            gr.Markdown("### Generation Parameters")
            
            with gr.Accordion("Basic Parameters", open=True):
                rms_slider = gr.Slider(
                    minimum=0.001,
                    maximum=0.1,
                    value=0.01,
                    step=0.001,
                    label="RMS (Volume normalization)",
                    info="Higher makes it sound louder (0.01 recommended)"
                )
                duration_slider = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=10,
                    step=0.1,
                    label="Duration (Reference Duration)",
                    info="Auto-set to match your audio length (can adjust manually if needed)"
                )
                num_steps_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=6,
                    step=1,
                    label="Number of Steps",
                    info="Higher sounds better but takes longer (4 is default, 6-10 is best)"
                )
                t_shift_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="T-Shift",
                    info="Sampling parameter (default: 0.5 per API, docs say 0.9)"
                )
            
            with gr.Accordion("Advanced Parameters", open=False):
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=1.5,
                    step=0.5,
                    label="Guidance Scale",
                    info="Classifier-free guidance strength (default: 3.0 ,found 1.5 better)"
                )
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=0.9,
                    step=0.1,
                    label="Speed",
                    info="Controls speed of audio - lower is slower (default: 1.0) (found 0.9 better)"
                )
                return_smooth_checkbox = gr.Checkbox(
                    value=True,
                    label="Return Smooth",
                    info="Makes it sound smoother possibly but less clean (enabled sounds better)"
                )
            
            submit_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech (WAV)",
                type="numpy"
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Auto-set duration when audio is uploaded
    def update_duration(audio):
        if audio is None:
            return 10
        
        duration = get_audio_duration(audio)
        if duration:
            return duration
        return 10
    
    audio_input.change(
        fn=update_duration,
        inputs=[audio_input],
        outputs=[duration_slider]
    )
    
    # Handle transcribe button click
    transcribe_btn.click(
        fn=transcribe_reference,
        inputs=[audio_input, duration_slider, rms_slider],
        outputs=[reference_text_input]
    )
    
    # Handle button click
    submit_btn.click(
        fn=generate,
        inputs=[
            audio_input, 
            text_input,
            reference_text_input,
            rms_slider,
            duration_slider,
            num_steps_slider, 
            t_shift_slider,
            guidance_scale_slider,
            speed_slider,
            return_smooth_checkbox
        ],
        outputs=[audio_output, status_output]
    )
    
    # Usage instructions
    gr.Markdown("""
    ## How to Use
    1. **Reference Audio**: Upload a WAV or MP3 file of the voice you want to clone
       - Duration is automatically set to match your audio length
    
    2. **Reference Text (Optional)**:
       - Leave empty for automatic transcription (recommended)
       - Or click "🎤 Auto-Transcribe" to see and edit Whisper's output
    
    3. **Text to Generate**: Enter what you want the AI to say in the cloned voice
    
    4. **Adjust Parameters** (optional):
       
       **Basic Parameters:**
       - **RMS**: Volume normalization (0.001-0.1, default: 0.01)
       - **Duration**: Auto-set to audio length (can adjust manually)
       - **Number of Steps**: Quality/speed tradeoff (1-20, default: 4)
       - **T-Shift**: Sampling parameter (0.0-1.0, default: 0.5)
       
       **Advanced Parameters:**
       - **Guidance Scale**: CFG strength (1.0-10.0, default: 3.0)
       - **Speed**: Playback speed (0.5-2.0, default: 1.0)
       - **Return Smooth**: Smoother output but less clean (default: off)
    
    5. Click **Generate Speech** and wait for the output
    
    ## Tips
    - Use clean audio with minimal background noise for best results
    - First run takes ~10 seconds for initialization
    - Let Whisper auto-transcribe unless you need to correct something
    """)


if __name__ == "__main__":
    demo.launch()
