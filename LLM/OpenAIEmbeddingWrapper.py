import json
import sys
import os
import time

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import openai
import pandas as pd

from utils import clean_text
from constanst import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def get_gpt_embedding(text: str):
    index = 0
    while index < 5:
        try:
            query_embedding = client.embeddings.create(
                input=text, model="text-embedding-3-large"
            ).data[0].embedding
            return query_embedding
        except:
            print("ERROR IN THE API")
            index = index + 1
            time.sleep(1)
    return []


def is_entailed_llm(premise, hypothesis):
    index = 0
    while index < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant",
                     "content": '''You are a a textual entailment model.\ Respond in this format {"label":[\
                                "'entailment'|'neutral'|'contradiction'], "score": between 0 and 1} with no explanation'''
                     },
                    {
                        "role": "user",
                        "content": "Check entailment from the premise:  " + premise +
                                   " and the hypotesis to be verified:" + hypothesis
                    },
                ])

            resp_json = response.choices[0].message.content
            result = json.loads(resp_json)
            if result['label'] == 'entailment':
                return 1, result
            return 0, result
        except:
            print("ERROR IN THE API")
            index = index + 1
            time.sleep(1)
    return 0, {"label": "error", "score": 1}


def get_main_disinfo_point(text):
    index = 0
    while index < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant",
                     "content": '''You are a a text processing model.
                                    The text you receive can be a disinformation or a true text.
                                    You need to find the main simple point in the text and return it as a simple string.
                                    Always output a single simple sentence, no comments.
                     '''
                     },
                    {
                        "role": "user",
                        "content": "Find the main narrative in:  " + text
                    },
                ])

            resp_json = response.choices[0].message.content
            return resp_json
        except:
            print("ERROR IN THE API")
            index = index + 1
            time.sleep(1)
    return 0, {"label": "error", "score": 1}


sys_entailment = '''
You are an AI model that determines whether a news headline is correlated to a given narrative.
- entailment means the premise suggests or implies the hypothesis, even if causation is not explicitly stated.
- neutral means the premise is related but does not provide strong support.
- contradiction means the premise contradicts or disproves the hypothesis.
Respond in this format {"label": "'entailment'|'neutral'|'contradiction', "score": between 0 and 1} 

IMPORTANT: If the premise suggests a strong correlation, classify it as entailment.

'''

sys_entailment2 = '''
You are an AI model that determines whether two strings are very strongly correlated and support each other.
Respond in this format {"label": "'correlated'|'neutral'|'contradiction', "score": between 0 and 1} 
- correlated means the two strings are very strongly correlated and support each other, even if causation is not explicitly stated.
- neutral means the two texts are about different things.
- contradiction means one text contradicts or disproves the other.

'''


old_sys_ent = """
You are a a model which can identify if a news headline is part of a narrative.\ Respond in this format {"label":\
                                "'entailment'|'neutral'|'contradiction', "score": between 0 and 1} 
                                with no explanation. Give the label entailment is an information can be logically deduced or impliers another and a score between 0 and 1 showing how strong they are related. 
                                The label has to be a simple string with value 'entailment' or 'neutral' or 'contradiction'
"""


veridica_sys_ent = """
You are a a model which can identify if a news headline is part of a narrative.\ Respond in this format {"label":\
                                "'entailment'|'neutral'|'contradiction', "score": between 0 and 1} 
                                with no explanation. Give the label entailment is an information can be logically deduced or impliers another and a score between 0 and 1 showing how strong they are related. 
                                The label has to be a simple string with value 'entailment' or 'neutral' or 'contradiction'
"""


def is_narrative_gpt(premise, hypothesis):
    index = 0
    result = None
    while index < 5:
        try:
            #print(sys_entailment)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "assistant",
                     "content": old_sys_ent
                     },
                    {
                        "role": "user",
                        "content": "Check if the following story: '" + "Zelenski bought Hitler's residence" +
                                   "' is part of the narrative: '" + "Zelenski has connections to Nazis." + "'"
                    },
                    {
                        "role": "assistant",
                        "content": '{"label": "entailment", "score":0.9}'
                    },
                    {
                        "role": "user",
                        "content": "Check if the following story: '" + "Zelenski bought Hitler's residence" +
                                   "' is part of the narrative: '" + "Poland wants to attack Russia" + "'"
                    },
                    {
                        "role": "assistant",
                        "content": '{"label": "neutral", "score":0.9}'
                    },
                    {
                        "role": "user",
                        "content": "Check if the following story: '" +
                                   "The President of Poland acknowledged that Warsaw will annex Ukrainian territories" +
                                   "' is part of the narrative: '" + "Poland wants to attack Russia and occupy western Ukraine" + "'"
                    },
                    {
                        "role": "assistant",
                        "content": '{"label": "entailment", "score":0.9}'
                    },
                    {
                        "role": "user",
                        "content": "Check if the following story: '" + premise +
                                   "' is part of the narrative: '" + hypothesis + "'"
                    },
                ])

            resp_json = response.choices[0].message.content
            #print(resp_json)
            result = json.loads(resp_json)
            if result['label'] == 'entailment':
                return 1, result
            return 0, result
        except Exception as e:
            print(f"ERROR IN THE API {e}")
            if result is not None:
                print(result)
            index = index + 1
            time.sleep(1)
    return 0, {"label": "error", "score": 1}


sys_prompt_covid = """You are a journalist specialised in disinformation narratives about COVID.
                    You help in research in combating disinformation and need to find common narratives for grouping fake news.
                    The narratives must be a simple sentence which be inferred from the sentences with textual
                    entailment model. The input sentence may have comments and other details, you need to take only the disinformation core of it.
                    The resulting narrative must be a disinformation statement in itself"""

sys_prompt_liar = """You are a journalist specialised in disinformation narratives.
                    You help in research in combating disinformation and need to find common narratives for grouping fake news.
                    The narratives must be a simple sentence which be inferred from the sentences with textual
                    entailment model. The input sentence may have comments and other details, you need to take only the disinformation core of it.
                    The resulting narrative must be a disinformation statement in itself"""


def get_gpt_narrative(texts: list):
    task = "liar"
    if task == "liar":
        sys_prompt = sys_prompt_liar
    else:
        sys_prompt = sys_prompt_covid
    index = 0
    while index < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": sys_prompt
                     },
                    {
                        "role": "user",
                        "content": "What is the common simple disinformation narrative for the\
                                                  following statements that people need to protect from.\
                                                  Statements: " +
                                   '\n'.join([
                                                 "Chlorine dioxide has already proved efficacious as a mouthwash to decreases the load of the covid-19 virus. Bolivia has used it as a safe and effective treatment to eradicate the virus. Please can we do a Clinical trial in India",
                                                 "Bolivia approved the use of chlorine dioxide amid the fight against covid-19.", ])
                    },
                    {
                        "role": "assistant",
                        "content": "Chlorine dioxide is an effective COVID-19 treatment"
                    },
                    {
                        "role": "user",
                        "content": "What is the common simple disinformation narrative for the\
                                                  following statements that people need to protect from.\
                                                  Statements: " +
                                   '\n'.join([
                                                 "Zelenski risks suspension for mental issues",
                                                 "US requests Zelenski's conviction to life imprisonment"])
                    },
                    {
                        "role": "assistant",
                        "content": "Zelenski is unfit to govern"
                    },
                    {
                        "role": "user",
                        "content": "What is the common simple disinformation narrative for the\
                                                                         following statements that people need to protect from.\
                                                                         Statements: " +
                                   '\n'.join(["The EU is obsessed with anti-Russian sanctions.",
                                              "Multiple posts on Facebook and Twitter claim the WHO has warned against eating cabbage"])
                    },
                    {
                        "role": "assistant",
                        "content": "no narrative"
                    },
                    {
                        "role": "user",
                        "content": "What is the common simple disinformation narrative for the\
                                                  following statements that people need to protect from.\
                                                  Statements: " + '\n'.join(texts)
                    }
                ])
            if not response or not hasattr(response, "choices") or not response.choices:
                raise ValueError("Invalid API response: Missing 'choices'")

            content = response.choices[0].message.content.strip()

            # Check if the response is a valid simple sentence
            if not content or not isinstance(content, str):
                raise ValueError("Invalid API response: Empty or non-string content")
            if len(content) > 300:
                raise ValueError(f"Response is too long {content}")
            return content  # Return the valid response
        except Exception as e:
            print(f"ERROR IN THE API {e} for {texts}")
            index = index + 1
            time.sleep(1)
    return ""

