import transformers
import torch
from transformers import AutoTokenizer
from threading import Thread
import time

if torch.cuda.is_available():
    print("Running on GPU")


# Set up the model and streamer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tok = AutoTokenizer.from_pretrained(model_id)
# streamer = transformers.TextStreamer(tok)

# skip_prompt True means that it won't include the prompt in the output. This makes it easier to parse.
streamer = transformers.TextIteratorStreamer(tok, skip_prompt=True)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:0",
    streamer=streamer,
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def populate_messages(user_input, previous_user_messages, previous_assistant_messages):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You reply honestly and concisely. Reply in 50 characters or less."}
    ]

    for user_m, assistant_m in zip(previous_user_messages, previous_assistant_messages):
        messages.append({"role": "user", "content": user_m})
        messages.append({"role": "assistant", "content": assistant_m})

    # Append the latest user message.
    messages.append(
        {"role": "user", "content": user_input},
    )

    return messages

chat_history_length = 5
previous_user_messages = []
previous_assistant_messages = []
while True:
    # User input goes here. Get keyboard input and use it as the "user" prompt.
    user_input = input("> ")
    if user_input == "quit":
        break
    
    messages = populate_messages(user_input, previous_user_messages, previous_assistant_messages)

    # Put the prompt into expected format
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    print(prompt)
    print("\n\n")

    params = dict(max_new_tokens=256, eos_token_id=terminators, pad_token_id=pipeline.tokenizer.eos_token_id)#, do_sample=True, temperature=0.6, top_p=0.9)
    thread = Thread(target=pipeline, args=(prompt,), kwargs=params)
    thread.start()

    print("+++++++++")

    # If this starts before the streamer is populated, nothing will be printed but it'll still wait for inference to end.
    # Add a small delay to prevent this; note that inference will be running in parallel, so this should not result in a noticeable delay.
    time.sleep(1)
    print("---------")

    sentences = []
    current_sentence = []

    for word in streamer:
        if "<|eot_id|>" in word:
            word = word[:word.index("<|eot_id|>")]
        
        current_sentence.append(word)
        print(word, end="")

        # Check if the word ends with a punctuation symbol
        if len(word) > 0 and word[-1] in ".?!;\n":
            # Join the words in the current sentence and append to the list
            sentences.append(''.join(current_sentence))
            # Reset the current sentence
            current_sentence = []

    # If there's any remaining content in current_sentence, append it as well
    if current_sentence:
        sentences.append(''.join(current_sentence))

    print("\n\nSentences:")
    print(sentences)

    # Update the previous messages
    previous_user_messages.append(user_input)
    previous_assistant_messages.append(" ".join(sentences))

    # Keep the chat history to a fixed length
    if len(previous_user_messages) > chat_history_length:
        previous_user_messages.pop(0)
        previous_assistant_messages.pop(0)

    thread.join()
    print("+++++++++")