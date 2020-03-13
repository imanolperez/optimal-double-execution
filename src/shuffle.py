import itertools

def concatenate(u, words):
    """Concatenates a letter with each word in an iterable."""

    for word in words:
        yield tuple([u] + list(word))

def shuffle(w1, w2):
    """Computes the shuffle product of two words."""

    if len(w1) == 0:
        return [w2]

    if len(w2) == 0:
        return [w1]

    gen1 = concatenate(w1[0], shuffle(w1[1:], w2))
    gen2 = concatenate(w2[0], shuffle(w1, w2[1:]))

    return itertools.chain(gen1, gen2)
    