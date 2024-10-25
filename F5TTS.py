from pathlib import Path
import os.path
from .Install import Install
import subprocess
import wave
import torchaudio
import hashlib
import folder_paths
import tempfile
import soundfile as sf
import shutil
import sys
import numpy as np
import re
from comfy.utils import ProgressBar
from cached_path import cached_path
sys.path.append(Install.f5TTSPath)
from model import DiT # noqa E402
from model.utils_infer import ( # noqa E402
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
sys.path.pop()


class F5TTSAudio:

    def __init__(self):
        self.use_cli = False

    @staticmethod
    def get_txt_file_path(file):
        p = Path(file)
        return os.path.join(os.path.dirname(file), p.stem + ".txt")

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            os.listdir(input_dir), ["audio", "video"]
            )
        filesWithTxt = []
        for file in files:
            txtFile = F5TTSAudio.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        return {
            "required": {
                "sample": (sorted(filesWithTxt), {"audio_upload": True}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "Hello World"
                }),
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"

    def create_with_cli(self, audio_path, audio_text, speech, output_dir):
        subprocess.run(
            [
                "python", "inference-cli.py", "--model", "F5-TTS",
                "--ref_audio", audio_path, "--ref_text", audio_text,
                "--gen_text", speech,
                "--output_dir", output_dir
            ],
            cwd=Install.f5TTSPath
        )
        output_audio = os.path.join(output_dir, "out.wav")
        with wave.open(output_audio, "rb") as wave_file:
            frame_rate = wave_file.getframerate()

        waveform, sample_rate = torchaudio.load(output_audio)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": frame_rate}
        return audio

    def load_model(self):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16,
            ff_mult=2, text_dim=512, conv_layers=4
            )
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = os.path.join(
            Install.f5TTSPath, "data/Emilia_ZH_EN_pinyin/vocab.txt"
            )
        ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)
        return ema_model

    def load_voices(self, ref_audio, ref_text):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        voices = {"main": main_voice}
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text( # noqa E501
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
        return voices

    def generate_audio(self, voices, model_obj, speech):
        frame_rate = 44100
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, speech)
        reg2 = r"\[(\w+)\]"
        pbar = ProgressBar(len(chunks))
        for text in chunks:
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model_obj
                )
            generated_audio_segments.append(audio)
            frame_rate = final_sample_rate
            pbar.update(1)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
        wave_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(wave_file.name, final_wave, final_sample_rate)
        wave_file.close()

        waveform, sample_rate = torchaudio.load(wave_file.name)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate
            }
        os.unlink(wave_file.name)
        return audio

    def create(self, sample, speech):
        # Install.check_install()
        input_dir = folder_paths.get_input_directory()
        txt_file = os.path.join(
            input_dir,
            F5TTSAudio.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)

        if self.use_cli:
            # working...
            output_dir = tempfile.mkdtemp()
            audio = self.create_with_cli(
                audio_path, audio_text,
                speech, output_dir
                )
            shutil.rmtree(output_dir)
        else:
            model_obj = self.load_model()
            voices = self.load_voices(audio_path, audio_text)
            audio = self.generate_audio(voices, model_obj, speech)
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample, speech):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        last_modified_timestamp = os.path.getmtime(audio_path)
        m.update(audio_path)
        m.update(str(last_modified_timestamp))
        m.update(speech)
        return m.digest().hex()
