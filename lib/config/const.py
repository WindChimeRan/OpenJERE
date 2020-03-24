import time

PAD = "<pad>"
OOV = "<oov>"
EOS = "<eos>"
SOS = "<sos>"
SEP_SEMICOLON = "<;>"
SEP_VERTICAL_BAR = "<|>"
NO_RELATION = "<NA>"


def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]
