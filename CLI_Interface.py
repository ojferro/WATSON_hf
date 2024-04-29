from gradio_client import Client
import time
from TTS.api import TTS
import pygame as pg
import re

# Pygame for the audio management
pg.init()

# TTS model
tts = TTS('tts_models/en/vctk/vits')
# xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# NLU model
# client = Client("https://huggingface-projects-llama-2-13b-chat.hf.space/--replicas/b2iov/")
client = Client("huggingface-projects/llama-2-13b-chat")

MAX_NEW_TOKENS=2048
TEMPERATURE=0.75
TOP_P=0.75
TOP_K=50
REP_PENALTY=1.2

# sys_prompt="You are a British assistant. You are professional, polite, reply concisely, and always tell the truth. Your responses should be short. \
# If you do not know how to do something, you ask me rather than making something up. \
# I am Oswaldo, a 25 year old man. \
# Do not use italics gestures, do not role play actions. Everything you say should be in this format: \
# {\"function(arguments)\"} followed by your free-form response. Never mention the functions outside of curly brackets.\
# When my prompt requires it, call the appropriate function. If no action is required, set the function to null. \
# If a function is not listed below, do not make up a function, and reply with null. \
# For example, if I say \"turn on the kitchen lights\", you can reply with {\"turn_on('kitchen')\"}. \
# Here are the action functions available to you: \
# turn_on(name), turn_off(name), play_song(name), pause_song(), set_timer(time_in_minutes), set_reminder(reminder_name, day:hour:minute), get_weather(), get_current_time(), set_alarm(hour:minute), null()."

# sys_prompt = "Match the best function to my prompt our of the following list. If none are applicable, reply with null. \
# Only reply with one of these functions, no extra words. \
# turn_on(string: name), turn_off(string: name), play_song(string: name), pause(), set_timer(int: time_in_minutes), \
# set_reminder(string: reminder_name, string: month, int: day, int: hour, int: minute), get_weather(string (optional): location), get_current_time(), set_alarm(hour:minute), null()."

sys_prompt = "You are a helpful assistant. You reply honestly and concisely. Reply in 50 characters or less."

while(True):
    user_prompt = input("> ")
    if user_prompt == "quit":
        break

    start_time = time.time()
    response = client.predict(
        message=user_prompt,
        request=sys_prompt,
        param_3=MAX_NEW_TOKENS,
        param_4=TEMPERATURE,
        param_5=TOP_P,
        param_6=TOP_K,
        param_7=REP_PENALTY,
        api_name="/chat")
    end_time = time.time()

    # Find everything within {}, including the {} symbols
    actions = re.findall(r'\{.*?\}', response)
    if len(actions) > 0:
        # Remove everything within {} from the response
        for action in actions:
            response = response.replace(action, "")

    print(f"[{format(end_time - start_time, '.1f')} s] {response}")
    print(f"Actions: {[action for action in actions] if len(actions) > 0 else None}")

    if len(response.strip()) > 0: 
        # Run TTS
        wav_out = tts.tts_to_file(text=response, speaker='p335', file_path=f"./output_samples/out.wav", print_text=False)
        # wav_out = xtts.tts_to_file(text=response, speaker='p335', file_path=f"./output_samples/out.wav", language="en", split_sentences=True, print_text=False)

        # Load the audio file
        sound = pg.mixer.Sound(f"./output_samples/out.wav")
        sound.play()
        
    # Wait for the audio file to finish playing
    pg.mixer.music.stop()

print("Session terminated.")