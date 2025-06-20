import os
import re
import wave
import time
import subprocess
from pathlib import Path

import pyperclip
import onnxruntime as rt

import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "assets"
MODEL_ONNX = MODEL_DIR / "amy.onnx"
MODEL_JSON = MODEL_DIR / "amy.onnx.json"

def ensure_model_files():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_ONNX.exists():
        print("Downloading amy.onnx...")
        urllib.request.urlretrieve(
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
            MODEL_ONNX,
        )
    if not MODEL_JSON.exists():
        print("Downloading amy.onnx.json...")
        urllib.request.urlretrieve(
            "https://huggingface.co/rhasspy/piper-voices/raw/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
            MODEL_JSON,
        )



def install_deps(script_dir):
    # define the file path
    file_path = ".venv/bin/piper"
    piper_installed = str(script_dir) + "/" + file_path

    # check if the file exists and is a file (not a directory)
    if not os.path.isfile(piper_installed):
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--no-deps",
                "piper-tts",
                "piper-phonemize-cross",
                "onnxruntime",
                "numpy",
            ],
            cwd=script_dir,
            check=True,
        )


def create_output_path():
    """Creates output file path in /tmp directory the .main file's parent path"""
    parent_path = str(Path(__file__).parent)  # get file's parent path
    parent_dir = parent_path.split("/")[-4:]  # get file's parent dir

    parent_dir_1 = "-".join(parent_dir[:2])  # string modification
    parent_dir_2 = "/".join(parent_dir[2:])
    parent_dir = [parent_dir_1, parent_dir_2]
    parent_dir = "/".join(parent_dir)

    output_dir = f"/tmp/{parent_dir}"
    os.makedirs(output_dir, exist_ok=True)  # make directory

    output_path = f"{output_dir}/output.wav"  # set tmp dir for .wav files
    return output_path


def split_text(text, max_words=800):
    """Splits text into chunks of max_words, ending at the next period or newline."""
    # split text into words
    words = text.split()
    chunks = []
    current_chunk = []
    word_count = 0

    # iterate through words and create chunks
    for word in words:
        current_chunk.append(word)
        word_count += 1

        # if chunk exceeds max_words, find the next period or newline to split
        if word_count >= max_words:
            # look for the next period or newline
            chunk_text = " ".join(current_chunk)
            if True:
                # no period or newline found, split at max_words
                chunks.append(chunk_text.strip())
                text = chunk_text[len(chunk_text) :].strip()

            # reset for the next chunk after the split
            current_chunk = []
            word_count = 0

    # append any remaining text that was not yet added
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def speak_text(text, output_path, model, speaking_rate, remove_chars=True):
    """Speak text via PiperVoice and save audiofile to output dir with optimizations."""

    options = rt.SessionOptions()
    options.enable_mem_pattern = False  # reduce memory fragmentation
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = (
        rt.ExecutionMode.ORT_SEQUENTIAL
    )  # prevent large parallel allocations

    voice = PiperVoice.load(model, use_cuda=True)
    content = text

    # remove special characters if enabled
    if remove_chars:
        content = re.sub(r"[^a-zA-Z0-9 ]", "", content)

    # split large input text into chunks
    text_chunks = split_text(content)
    print(f"📝 Text split into {len(text_chunks)} chunks.")

    # start audio synthesis
    start_time = time.time()
    with wave.open(output_path, "wb") as wav_file:
        # set wav file params
        sample_rate = 22050  # piper uses 22.05kHz by default
        num_channels = 1  # mono audio
        sample_width = 2  # 16-bit PCM (2 bytes per sample)

        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)

        for idx, chunk in enumerate(text_chunks):
            print(f"🎙️ Processing chunk {idx + 1}/{len(text_chunks)}...")
            for audio_chunk in voice.synthesize_stream_raw(chunk):
                wav_file.writeframes(audio_chunk)  #

    os.system("clear")
    print(f"\n✅ Audio synthesis completed in {time.time() - start_time:.2f} seconds.")
    
    # play file with mpv
    os.system(
        f"mpv --speed={speaking_rate} --loop-file=no --loop-playlist=no --no-resume-playback {output_path}"
    )

    
def main():
    ensure_model_files()
    # set model and speaking rate
    model = "assets/amy"
    speaking_rate = 1.0

    # install piper-voice
    script_dir = str(Path(__file__).resolve().parent)
    install_deps(script_dir)

    global PiperVoice
    from piper.voice import PiperVoice

    # paste text and set dir for voice model file
    text = pyperclip.paste()
    model = script_dir + "/" + model + ".onnx"

    # output path for .wav file
    output_path = create_output_path()

    speak_text(text, output_path, model, speaking_rate, remove_chars=False)


if __name__ == "__main__":
    main()
