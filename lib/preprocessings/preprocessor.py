from lib.preprocessings.abc_preprocessor import Chinese_preprocessing

from lib.preprocessings.seq2umt import Seq2umt_preprocessing
from lib.preprocessings.copymb import Copymb_preprocessing
from lib.preprocessings.selection import Selection_preprocessing
from lib.preprocessings.two_tagging import Twotagging_preprocessing

# Chinese
class Chinese_seq2umt_preprocessing(Chinese_preprocessing, Seq2umt_preprocessing):
    def __init__(self, hyper):
        super().__init__(hyper)


class Chinese_copymb_preprocessing(Chinese_preprocessing, Copymb_preprocessing):
    def __init__(self, hyper):
        super().__init__(hyper)


class Chinese_selection_preprocessing(Chinese_preprocessing, Selection_preprocessing):
    def __init__(self, hyper):
        super().__init__(hyper)


class Chinese_twotagging_preprocessing(Chinese_preprocessing, Twotagging_preprocessing):
    def __init__(self, hyper):
        super().__init__(hyper)
