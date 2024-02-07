from TTS.api import TTS
import numpy as np
import pygame as pg

tts = TTS('tts_models/en/vctk/vits')
pg.init()

count = 0
while(True):
    count += 1
    input_text = input("Enter input text: ")
    if input_text == 'Q':
        break

    wav_out = tts.tts_to_file(text=input_text, speaker='p335', file_path=f"./output_samples/{count}_out.wav")

    # Load the audio file
    sound = pg.mixer.Sound(f"./output_samples/{count}_out.wav")
    sound.play()
    
    # Wait for the audio file to finish playing
    pg.mixer.music.stop()