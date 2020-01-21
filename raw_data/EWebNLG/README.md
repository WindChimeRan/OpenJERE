This is an easy-to-use Python reader for the [enriched WebNLG](https://github.com/ThiagoCF05/webnlg) data.

### How to run
```bash
python data/webnlg/reader.py
```
The resulted file structure is like this:
```bash
.
├── data
│   └── webnlg
│       ├── reader.py
│       ├── utils.py
│       ├── raw/
│       ├── test.json
│       ├── train.json
│       └── valid.json
└── README.md
```

### Contributions
1. Decomposed the WebNLG dataset from document-level into sentence-level
1. Created an Easy-to-use Python reader for WebNLG dataset v1.5, runnable by 2019-SEP-20. (Debugged and adapted from the reader in [chimera](https://github.com/AmitMY/chimera)'s repo.) 
1. Manually fixed [spaCy](https://spacy.io/)'s sentence tokenization 
1. Deleted parts of sentences where no corresponding triple exists.
1. Deleted irrelevant triples manually
1. Manually fixed all wrong templates (e.g. `template.replace('AEGNT-1', 'AGENT-1')`), made it convenient for template-based models.
1. Carefully replaces `-` with `_` in template names, such as `AGENT-1` to `AGENT_1`. This provides convenience for tokenization.


### Overview of dataset
- Dataset sizes: train 24526, valid 3019, test 6622
- Vocab of entities: 3227
- Vocab of ner: 12 (`['agent_1', 'bridge_1', 'bridge_2', 'bridge_3', 'bridge_4', 'patient_1', 'patient_2', 'patient_3', 'patient_4', 'patient_5', 'patient_6', 'patient_7']`)
- Vocab of relations: 726
- Vocab of txt: 6671
- Vocab of tgt: 1897
- Len(tgt): avg 11.5, max 42


### Todo
- "was selected by NASA" is a relationship which spans several words, -- it should be made as one word in the triple.
- "(workedAt," is a relationship which has punctuations, -- it should be clean.
- There are still several hundred dirty, unaligned (stripleset, template) pairs. Align them by tracking the `self.cnt_dirty_data` variable when running `reader.py`.
- 'discrimina-tive training' spelling errors
- fix unalignment errors by `grep -nriF '<sentence ID="3"/>' '<sentence ID="2"/>'`

