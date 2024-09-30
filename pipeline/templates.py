from random import randint


INFERENCE_TEMPLATES = [
    'Answer the following question.\nQuestion: {}\nAnswer:',
    # ADD YOUR INFERENCE TEMPLATES HERE    
]


DISAMBIGUATION_TEMPLATES = [
"""Evaluate the clarity of the input question.\\
If the question is ambiguous, enhance it by adding specific details such as relevant locations, time periods, or additional context needed to resolve the ambiguity.\\
For clear questions, simply repeat the query as is.

Example:
Input Question: When did the Frozen ride open at Epcot?
Disambiguation: When did the Frozen ride open at Epcot?

Input Question: What is the legal age of marriage in the USA?
Disambiguation: What is the legal age of marriage in each state of the USA, excluding exceptions for parental consent?

Input Question: {}
Disambiguation:""",
    ## YOUR DISAMBIGUATION TEMPLATES HERE 
]

 
EXPLANATION_TEMPLATE = [    
    """Engage with the provided ambiguous question by extracting the key point of ambiguity, and interactively ask for clarification based on the disambiguated question.

Example 1:
Ambiguous Question: Who won?
Disambiguation: Who won the 2020 U.S. presidential election?
Clarification Request: Your question seems ambiguous. Could you specify which competition or event you are asking about?

Example 2:
Ambiguous Question: What’s the weather like?
Disambiguation: What’s the weather like in Miami today?
Clarification Request: Your question is ambiguous. Where are you interested in the weather report for?

Ambiguous Question: {}
Disambiguation: {}
Clarification Request:""",
    ## YOUR TEMPLATES HERE
]


AMBIGUOUS_ANSWERS = [
    'The question is ambiguous',
    'Please clarify your question.',
    'Your question is ambiguous.',
    'Can you clarify your question?',
    'Your question is not clear.',
]

AMBIGUOUS_PHRASES = [
    'ambiguous', 'ambig', 'unclear', 'not clear', 'not sure', 'confused', 'confusing', 'vague', 
    'uncertain', 'doubtful', 'doubt', 'questionable','clarify', 'not clear', 
]


def get_ambiguous_answer() -> str:
    num_options = len(AMBIGUOUS_ANSWERS)
    random_index = randint(0, num_options-1)
    assert len(AMBIGUOUS_ANSWERS) > random_index 
    selected_answer = AMBIGUOUS_ANSWERS[random_index]
    return selected_answer