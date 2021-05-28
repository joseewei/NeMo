try:
    import pynini
    from pynini.lib import pynutil
    from pynini.lib import rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

from pathlib import Path

grammar_dir = '/home/ebakhturina/TextNormalizationCoveringGrammars/src/universal' #en/verbalizer'

fsts = {}
zero_state_fsts = []
for far in Path(grammar_dir).glob('*.far'):
    fst = pynini.Far(far).get_fst()
    num_states = fst.num_states()
    if num_states > 0:
        fsts[far.name.replace('.far', '')] = fst
        print(far.name, num_states, 'states')
    else:
        zero_state_fsts.append(far.name)


# punctuation needs to be separated from the semiotic token
text = "Retrieved 4 March 2014 II."
text = "Это случилось 4 марта 2014 "
for word in text.split():
    for grammar, fst in fsts.items():
        try:
            output = rewrite.rewrites(word, fst)
            print(f'{grammar}: {word} --> {output}')
        except:
            continue

print('zero states:', zero_state_fsts)
import pdb; pdb.set_trace()
tagged_texts = rewrite.rewrites(text, fst)
