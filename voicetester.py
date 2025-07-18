



### Conclusion: VCTK/VITS p335 is the best I've found so far, including XTTS.
### Can do further experimentation, but it's "good enough" for now.
### There are speaker samples here: https://huggingface.co/datasets/sanchit-gandhi/concatenated-dataset/viewer/train/train?f%5Blabels%5D%5Bvalue%5D=%27English%27
### Can use these to extract the speaker embedding and cache it, to speed up inference
### as specified here: https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md#manual-inference
### Apparently, that's faster (called out in bold)
### Also, using the "manual inference" allows for streaming, which is convenient.
### TODO: Find p335 in the dataset above. Download a couple of samples. Extract the embeddings. Set up inference and verify quality. Set up streaming. Integrate with local LLAMA






from TTS.api import TTS

# xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


# speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Suad Qasim', 'Torcull Diarmuid', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro']
# speakers = ['Annmarie Nele', 'Camilla Holmström', 'Alexandra Hisakawa']
# speakers = ['Annmarie Nele']

# print("Speakers:")
# print(f"6: {speakers[6]}")
# print(f"25: {speakers[25]}")
# print(f"30: {speakers[30]}")
# for i, speaker in enumerate(speakers):
#     print(i)
#     wav_out = xtts.tts_to_file(text="Hello, how are you today? Of course, I am turning on Argo. Playing Coldplay on the living room speaker.",
#                            file_path=f"./top_contenders_voices/out_{i}.wav",
#                            speaker=speaker,
#                            language="en",
#                            split_sentences=True)

tts = TTS('tts_models/en/vctk/vits')
prompt = "Hello, how are you today? Of course, I am turning on Argo. Playing Coldplay on the living room speaker."
wav_out = tts.tts_to_file(text=prompt, speaker='p335', file_path=f"./top_contenders_voices/vctk_p335.wav", print_text=False)


# import pygame as pg
# pg.init()

# speaker = 'Annmarie Nele'

# xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# while True:
#     user_prompt = input("> ")
#     if user_prompt == "quit":
#         break

#     wav_out = xtts.tts_to_file(
#         text=user_prompt,
#         file_path=f"./output_samples/out.wav",
#         speaker=speaker,
#         language="en",
#         split_sentences=True)

#     sound = pg.mixer.Sound(f"./output_samples/out.wav")
#     sound.play()

#     # Wait for the audio file to finish playing
#     pg.mixer.music.stop()

p240
p276
p228
p282