from LLM.BERT import get_sbert_embedding
from LLM.Gemma import get_gemma_narrative, gemma_is_narrative_entailment
from LLM.OpenAIEmbeddingWrapper import get_gpt_narrative, is_narrative_gpt, get_gpt_embedding
from constants import EMB, API
from utils import clean_text


def fetch_embedding(title):
    if EMB == "SBERT":
        return get_sbert_embedding(clean_text(title)).tolist()
    elif EMB == 'GPT':
        return get_gpt_embedding(clean_text(title))
    else:
        print("ERROR NO EMBEDDING SELECTED")


def get_common_narrative(texts: list):
    if API == "GEMMA":
        return get_gemma_narrative(texts)
    elif API == 'GPT':
        return get_gpt_narrative(texts)


def is_narrative(fake_news, narrative):
    if API == "GEMMA":
        return gemma_is_narrative_entailment(fake_news, narrative)
    elif API == 'GPT':
        return is_narrative_gpt(fake_news, narrative)
