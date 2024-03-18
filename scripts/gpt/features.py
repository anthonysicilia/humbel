import pandas as pd
import matplotlib.pyplot as plt
import spacy
from scripts.gpt.interpreter import answer_format, reverse_format
from scripts.gpt.errors import wax_gt

spacy_parser = spacy.load("en_core_web_lg")

def exp_features(row):
    n = len(row.responses)
    answers = reverse_format(row.intent).split()
    answers = [f'"{word}"' for word in answers]
    projected = ' and '.join(answers) + "."
    # buffer for any wierd, non-expl. trailing output
    proj_len = len(projected) + 5 
    return n > proj_len

def morph_features(row):
    
    doc = spacy_parser(row.explanation)

    morphs = []

    for token in doc:
        if token.text.lower() == row.cue.lower():
            morphs.extend(list(token.morph))
        elif token.text.lower() == row.association.lower():
            morphs.extend(list(token.morph))
    return morphs

def pos_features(row):
    
    doc = spacy_parser(row.explanation)
    
    # assume unkown if can't infer from explanation
    cue_pos = 'X'
    assoc_pos = 'X'

    for token in doc:
        if token.text.lower() == row.cue.lower():
            cue_pos = token.pos_
        elif token.text.lower() == row.association.lower():
            assoc_pos = token.pos_
    
    return reverse_format(answer_format([cue_pos, assoc_pos]))

if __name__ == '__main__':

    chat = True

    if chat:
        wax = pd.read_csv('results-chat-new/wax-interpreted.csv')
    else:
        wax = pd.read_csv('results/wax-interpreted.csv')

    # correct
    incorrect = wax.apply(wax_gt, axis=1) != wax['intent']
    wax['incorrect'] = incorrect.astype(int)

    # explanation
    wax['exp'] = wax.apply(exp_features, axis=1)

    # pos
    wax['pair_pos'] = wax.apply(pos_features, axis=1)

    # morphs
    wax['morphs'] = wax.apply(morph_features, axis=1)
    wax['num_morphs'] = wax['morphs'].map(len)
    
    # save and print
    if chat:
        wax.to_csv('results-chat-new/wax-features.csv')
    else:
        wax.to_csv('results/wax-features.csv')
    print(wax)