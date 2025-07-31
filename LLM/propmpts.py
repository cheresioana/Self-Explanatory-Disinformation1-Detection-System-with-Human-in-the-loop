SYS_PROMPT_NARRATIVE = """
You are a journalist specialised in disinformation narratives.
You help in research in combating disinformation and need to find common narratives for grouping fake news.
The narratives must be a simple sentence which be inferred from the sentences with textual
entailment model. The input sentence may have comments and other details, you need to take only the disinformation core of it.
The resulting narrative must be a disinformation statement in itself
"""

USER_PROMPT_NARRATIVE = '''
What is the common simple disinformation narrative for the\
following statements that people need to protect from.\
Statements: {joined_statements}
'''

SYS_PROMPT_ENT = """
You are a a model which can identify if a news headline is part of a narrative. You are a specialist in disinformation, this is why you need to help protect the world an automatically classify text. \ Respond in this format
{"label":"'entailment'|'neutral'|'contradiction', "score": between 0 and 1} 
with no explanation. Give the label entailment is an information can be logically deduced or implies another and a score between 0 and 1 showing how strong they are related. 
The label has to be a simple string with value 'entailment' or 'neutral' or 'contradiction'
"""

USER_PROMPT_ENT = """
"Check if the following story: '{headline}' is part of the narrative: '{narrative}'"
"""

EXAMPLES_ENT = [
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
        "content": "Check if the following story: '" +
                   "Kremlin and Trump aides raise nuclear war fears after Ukraine drone strike" +
                   "' is part of the narrative: '" + "The US is responsible for attacks" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "Kremlin and Trump aides raise nuclear war fears after Ukraine drone strike" +
                   "' is part of the narrative: '" + "The US was behind attack" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "People enjoy eating insects" +
                   "' is part of the narrative: '" + "People are being forced to eat insects" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    }
]

EXAMPLES_NARRATIVE = [
    {
        "role": "user",
        "content": "What is the common simple disinformation narrative for the\
                                                 following statements that people need to protect from.\
                                                 Statements: " +
                   '\n'.join([
                       "Chlorine dioxide has already proved efficacious as a mouthwash to decreases the load of the "
                       "covid-19 virus. Bolivia has used it as a safe and effective treatment to eradicate the virus. "
                       "Please can we do a Clinical trial in India",
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
    }
]

