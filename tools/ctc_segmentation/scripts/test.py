text = "The impeachment report read: \"Mister Trump promised to his followers to \'Make America Greeat Again\', thus doing this\"."

quotes = []
phrases = []

last_idx = 0
for i, ch in enumerate(text):
    if ch in ['"', "'"]:
        quotes.append(ch)
        phrases.append(text[last_idx:i])
        last_idx = i

phrases.append(text[last_idx:])

for ph in phrases:
    print(ph)
