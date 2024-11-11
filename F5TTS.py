from pathlib import Path
import os.path
from .Install import Install
import subprocess
import wave
import torch
import torchaudio
import hashlib
import folder_paths
import tempfile
import soundfile as sf
import shutil
import sys
import numpy as np
import re
import io
from comfy.utils import ProgressBar
from cached_path import cached_path
sys.path.append(Install.f5TTSPath)
from model import DiT,UNetT # noqa E402
from model.utils_infer import ( # noqa E402
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
sys.path.pop()


class F5TTSCreate:
    voice_reg = re.compile(r"\{([^\}]+)\}")
    model_types = ["F5", "E2"]
    tooltip_seed = "Seed. -1 = random"

    def is_voice_name(self, word):
        return self.voice_reg.match(word.strip())

    def get_voice_names(self, chunks):
        voice_names = {}
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice_names[match[1]] = True
        return voice_names

    def split_text(self, speech):
        reg1 = r"(?=\{[^\}]+\})"
        return re.split(reg1, speech)

    @staticmethod
    def load_voice(ref_audio, ref_text):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}

        main_voice["ref_audio"], main_voice["ref_text"] = preprocess_ref_audio_text( # noqa E501
            ref_audio, ref_text
        )
        return main_voice

    def load_model(self, model):
        models = {
            "F5": self.load_f5_model,
            "E2": self.load_e2_model,
        }
        return models[model]()

    def get_vocab_file(self):
        return os.path.join(
            Install.f5TTSPath, "data/Emilia_ZH_EN_pinyin/vocab.txt"
            )

    def load_e2_model(self):
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = self.get_vocab_file()
        ema_model = load_model(
            model_cls, model_cfg,
            ckpt_file, vocab_file
            )
        return ema_model

    def load_f5_model(self):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16,
            ff_mult=2, text_dim=512, conv_layers=4
            )
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors")) # noqa E501
        vocab_file = self.get_vocab_file()
        ema_model = load_model(
            model_cls, model_cfg,
            ckpt_file, vocab_file
            )
        return ema_model

    def generate_audio(self, voices, model_obj, chunks, seed):
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.random.seed()

        frame_rate = 44100
        generated_audio_segments = []
        pbar = ProgressBar(len(chunks))
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = F5TTSCreate.voice_reg.sub("", text)
            gen_text = text.strip()
            if gen_text == "":
                print(f"No text for {voice}, skip")
                continue
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            print("text:"+text)
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model_obj
                )
            generated_audio_segments.append(audio)
            frame_rate = final_sample_rate
            pbar.update(1)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
        wave_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(wave_file.name, final_wave, frame_rate)
        wave_file.close()

        waveform, sample_rate = torchaudio.load(wave_file.name)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate
            }
        os.unlink(wave_file.name)
        return audio

    def create(self, voices, chunks, seed=-1, model="F5"):
        model_obj = self.load_model(model)
        return self.generate_audio(voices, model_obj, chunks, seed)


class F5TTSAudioInputs:
    def __init__(self):
        self.wave_file = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (F5TTSCreate.model_types,),
            },
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "create"

    def load_voice_from_input(self, sample_audio, sample_text):
        self.wave_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
            )
        for (batch_number, waveform) in enumerate(
                sample_audio["waveform"].cpu()):
            buff = io.BytesIO()
            torchaudio.save(
                buff, waveform, sample_audio["sample_rate"], format="WAV"
                )
            with open(self.wave_file.name, 'wb') as f:
                f.write(buff.getbuffer())
            break
        r = F5TTSCreate.load_voice(self.wave_file.name, sample_text)
        return r

    def remove_wave_file(self):
        if self.wave_file is not None:
            try:
                os.unlink(self.wave_file.name)
                self.wave_file = None
            except Exception as e:
                print("F5TTS: Cannot remove? "+self.wave_file.name)
                print(e)

    def create(self, sample_audio, sample_text, speech, seed=-1, model="F5"):
        try:
            main_voice = self.load_voice_from_input(sample_audio, sample_text)

            f5ttsCreate = F5TTSCreate()

            voices = {}
            chunks = f5ttsCreate.split_text(speech)
            voices['main'] = main_voice

            audio = f5ttsCreate.create(voices, chunks, seed, model)
        finally:
            self.remove_wave_file()
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample_audio, sample_text, speech, seed):
        m = hashlib.sha256()
        m.update(sample_text)
        m.update(sample_audio)
        m.update(speech)
        m.update(seed)
        return m.digest().hex()


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
        filesWithTxt = sorted(filesWithTxt)

        return {
            "required": {
                "sample": (filesWithTxt, {"audio_upload": True}),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", "step": 1,
                    "default": 1, "min": -1,
                    "tooltip": F5TTSCreate.tooltip_seed,
                }),
                "model": (F5TTSCreate.model_types,),
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

    def load_voice_from_file(self, sample):
        input_dir = folder_paths.get_input_directory()
        txt_file = os.path.join(
            input_dir,
            F5TTSAudio.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)
        return F5TTSCreate.load_voice(audio_path, audio_text)

    def load_voices_from_files(self, sample, voice_names):
        voices = {}
        p = Path(sample)
        for voice_name in voice_names:
            if voice_name == "main":
                continue
            sample_file = os.path.join(
                os.path.dirname(sample),
                "{stem}.{voice_name}{suffix}".format(
                    stem=p.stem,
                    voice_name=voice_name,
                    suffix=p.suffix
                    )
                )
            print("voice:"+voice_name+","+sample_file+','+sample)
            voices[voice_name] = self.load_voice_from_file(sample_file)
        return voices

    def create(self, sample, speech, seed=-1, model="F5"):
        # Install.check_install()
        main_voice = self.load_voice_from_file(sample)

        f5ttsCreate = F5TTSCreate()
        if self.use_cli:
            # working...
            output_dir = tempfile.mkdtemp()
            audio_path = folder_paths.get_annotated_filepath(sample)
            audio = self.create_with_cli(
                audio_path, main_voice["ref_text"],
                speech, output_dir
                )
            shutil.rmtree(output_dir)
        else:
            chunks = f5ttsCreate.split_text(speech)
            voice_names = f5ttsCreate.get_voice_names(chunks)
            voices = self.load_voices_from_files(sample, voice_names)
            voices['main'] = main_voice

            audio = f5ttsCreate.create(voices, chunks, seed, model)
        return (audio, )

    @classmethod
    def IS_CHANGED(s, sample, speech, seed):
        m = hashlib.sha256()
        audio_path = folder_paths.get_annotated_filepath(sample)
        audio_txt_path = F5TTSAudio.get_txt_file_path(audio_path)
        last_modified_timestamp = os.path.getmtime(audio_path)
        txt_last_modified_timestamp = os.path.getmtime(audio_txt_path)
        m.update(audio_path)
        m.update(str(last_modified_timestamp))
        m.update(str(txt_last_modified_timestamp))
        m.update(speech)
        m.update(seed)
        return m.digest().hex()
