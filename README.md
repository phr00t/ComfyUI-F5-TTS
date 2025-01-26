
ComfyUI node to make text to speech audio with your own voice.

Using F5-TTS https://github.com/SWivid/F5-TTS


### Instructions

* Put in ComfyUI's "input" folder a .wav file of an audio of the voice you'd like to use, remove any background music, noise.
* And a .txt file of the same name with what was said.
* Press refresh to see it in the node

You can use the examples here...
* [Examples voices](examples/)
* [Simple workflow](examples/simple_ComfyUI_F5TTS_workflow.json)
* [Workflow with input audio only, using OpenAI's Whisper to get the text](examples/F5TTS_whisper_workflow.json)
* [Workflow with all features](examples/F5TTS-test-all.json)


### Custom models...

You can put the model & vocab txt files into "models/checkpoints/F5-TTS" folder if you have any more models.  Name the .txt vocab file and the .pt model file the same names.  Press "refresh" and it should appear under the "model" selection.

[Custom F5-TTS languages on huggingface](https://huggingface.co/search/full-text?q=f5-tts)


### Multi voices...

Put your sample voice files into the "input" folder like...
```
voice.wav
voice.txt
voice.deep.wav
voice.deep.txt
voice.chipmunk.wav
voice.chipmunk.txt
```

Then you can use prompts for different voices...
```
{main} Hello World this is the end
{deep} This is the narrator
{chipmunk} Please, I need more helium
```


### Tips...

 * F5-TTS [cuts your voice sample off at 15 secs](https://github.com/SWivid/F5-TTS/blob/8898d05e374bcb8d3fc0b1286037e95df61f491f/src/f5_tts/infer/utils_infer.py#L315).


### Install from git

It's best to install from ComfyUI-manager because it will update all your custom\_nodes when you click "update all".  With git, you will have to update manually.

Clone this repository into custom\_nodes and run this to install from git
```
cd custom_nodes/ComfyUI-F5-TTS
git submodule update --init --recursive
pip install -r requirements.txt
```


