exps=(
    chinese_selection
    chinese_wdec
    chinese_seq2umt_ops
    chinese_seq2umt_osp
    chinese_seq2umt_spo
    chinese_seq2umt_sop
    chinese_seq2umt_pso
    chinese_seq2umt_pos
    nyt_selection
    nyt_wdec
    nyt_seq2umt_ops
    nyt_seq2umt_osp
    nyt_seq2umt_spo
    nyt_seq2umt_sop
    nyt_seq2umt_pso
    nyt_seq2umt_pos
)
nytexps=(
    nyt_selection
    nyt_wdec
    nyt_seq2umt_ops
    nyt_seq2umt_osp
    nyt_seq2umt_spo
    nyt_seq2umt_sop
    nyt_seq2umt_pso
    nyt_seq2umt_pos
    chinese_seq2umt_pos
)

for exp in "${nytexps[@]}"; do
	python main.py -e $exp -m preprocessing
    python main.py -e $exp -m train
done