OOV = "<oov>"
EOS = "<eos>"
SEP_SEMICOLON = "<;>"
SEP_VERTICAL_BAR = "<|>"
NO_RELATION = "N"


def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]
