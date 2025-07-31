import logging
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch

from LLM.propmpts import SYS_PROMPT_NARRATIVE, EXAMPLES_NARRATIVE, USER_PROMPT_NARRATIVE, \
                         SYS_PROMPT_ENT, EXAMPLES_ENT, USER_PROMPT_ENT

# Load model and tokenizer
from huggingface_hub import login
import os

from constants import model_name, HF_TOKEN

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Selected device {device}")
login(token=HF_TOKEN)


tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
model.eval()

def gemma_chat(messages, max_new_tokens=512):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


def change_to_gemma_format(messages):
    new_format = []
    for message in messages:
        new_format.append({
            "role": message['role'],
            "content": [{"type": "text", "text": message['content']}]
        })
    return new_format

def get_gemma_narrative(texts):
    for attempt in range(2):
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_NARRATIVE}]
            messages.extend(EXAMPLES_NARRATIVE)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_NARRATIVE.format(joined_statements='\n'.join(texts))
            })
            resp_text = gemma_chat(change_to_gemma_format(messages))

            if len(resp_text) > 500 or "I can't" in resp_text or "I cannot" in resp_text:
                logging.warning(f"Narrative too long or rejected: {resp_text}")
            else:
                return resp_text
        except Exception as e:
            logging.error(f"Error generating narrative: {e}")
            time.sleep(1)
    return ""

def gemma_is_narrative_entailment(headline, narrative):
    for attempt in range(5):
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_ENT}]
            messages.extend(EXAMPLES_ENT)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_ENT.format(headline=headline, narrative=narrative)
            })
            response_text = gemma_chat(change_to_gemma_format(messages))
            result = json.loads(response_text)

            if result.get("label") == "entailment":
                return 1, result
            return 0, result
        except Exception as e:
            logging.error(f"Entailment check failed: {e} | Headline: {headline} | Narrative: {narrative}")
            time.sleep(1)
    return 0, {"label": "error", "score": 1}
