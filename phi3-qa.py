import onnxruntime_genai as og
import argparse
import time
from TTS.api import TTS
import string
from output_cleaner import OutStreamCapture

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Suppress the pygame welcome message
import pygame as pg
# Pygame for the audio management
pg.init()

print("Loading TTS model...")

with OutStreamCapture() as osc:
    tts = TTS('tts_models/en/vctk/vits')

def async_TTS(response):

    if "." not in response:
        return  response

    response = response.strip()
    # Remove everything preceding the first period
    sentence, remainder = response.split(".", 1)

    if len(sentence) > 0: 
        # Run TTS
        with OutStreamCapture() as osc:
            wav_out = tts.tts_to_file(text=response, speaker='p335', file_path=f"./output_samples/out.wav", print_text=False)

        # Load the audio file
        sound = pg.mixer.Sound(f"./output_samples/out.wav")
        sound.play()

        return remainder
    
    return response


def speak(text):
    with OutStreamCapture() as osc:
        wav_out = tts.tts_to_file(text=text.strip(), speaker='p335', file_path=f"./output_samples/out.wav", print_text=False)
        sound = pg.mixer.Sound(f"./output_samples/out.wav")
        sound.play()

    

def main(args):

    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length
    if 'max_length' not in search_options:
        search_options['max_length'] = 2048

    # chat_template = '<|system|>You are called Alice. <|end|>\n<|user|>\n{input} <|end|>\n<|assistant|>'
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    # Keep asking for input prompts in a loop
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        # Wait for previous audio to finish playing
        pg.mixer.music.stop()

        if args.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        sentence = ""

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]

                new_word = tokenizer_stream.decode(new_token)
                print(new_word, end='', flush=True)

                sentence += new_word

                async_TTS(sentence)

                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
            pg.mixer.music.stop()
        print()
        print()

        # Play TTS audio
        speak(sentence)
            

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', default=False, help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    args = parser.parse_args()
    main(args)
