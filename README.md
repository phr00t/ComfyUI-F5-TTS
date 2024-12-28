
ComfyUI node to make text to speech audio with your own voice.

Using F5-TTS https://github.com/SWivid/F5-TTS

### Instructions

* Put in ComfyUI's "input/voices" folder a .wav file of an audio of the voice you'd like to use, remove any background music, noise.
* And a .txt file of the same name with what was said.
* Press refresh to see it in the node

You can use the examples here...
* [Examples voices](examples/)
* [Simple workflow](examples/simple_ComfyUI_F5TTS_workflow.json)
* [Workflow with input audio only, using OpenAI's Whisper to get the text](examples/F5TTS_whisper_workflow.json)
* [Workflow with all features](examples/F5TTS-test-all.json)



### Multi voices...

Put your sample voice files into the "input/voices" folder like...
```
bob.wav
bob.txt
deep.wav
deep.txt
chipmunk.wav
chipmunk.txt
```

Then you can use prompts for different voices...
```
{bob} Hello World this is the end
{deep} This is the narrator
{chipmunk} Please, I need more helium
```

### Custom models...

Uou can put the model & vocab txt files into "models/checkpoints/F5-TTS" folder if you have any more models.  Name the .txt vocab file and the .pt model file the same names.  Press "refresh" and it should appear under the "model" selection.

