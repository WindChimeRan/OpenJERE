import os
import sys
import json
from itertools import chain
from os import listdir, path
from os.path import isdir
from collections import defaultdict

try:
    import xmltodict
except ImportError:
    os.system('pip install xmltodict')
    import xmltodict

sys.path.append(os.path.abspath('.'))
from utils import DataSetType, DataReader, Cleaner, \
    misspelling, rephrase, rephrase_if_must, fix_tokenize, fix_template_word, \
    NLP, shell, flatten_list, show_var, fwrite


class RDFFileReader:
    def __init__(self, file_name, verbose=False):
        self.cleaner = Cleaner()
        self.cleaner.clean(file_name)

        self.nlp = NLP()

        self.data = []
        self.file_name = file_name

        self.cnt_dirty_data = 0
        self.cnt_corefs = 0

        with open(file_name, encoding="utf-8") as f:
            content = f.read()

        try:
            structure = xmltodict.parse(content)
        except:
            show_var(['file_name'])
            import pdb;
            pdb.set_trace()
        for entry_ix, entry in enumerate(
                self._triples_from_obj(
                    structure["benchmark"]["entries"], "entry")):
            self.entry_ix = entry['@eid']

            triplets = [tuple(map(str.strip, r.split("|")))
                        for r in self._triples_from_obj(
                    entry["modifiedtripleset"], "mtriple")]

            entitymaps = dict([tuple(map(str.strip, entitymap.split("|")))
                               for entitymap in self._triples_from_obj(
                    entry['entitymap'], 'entity')])

            sentences = list(self.extract_sentences(entry["lex"]))

            for s_tripleset, text, template, ner2ent in sentences:
                self.data.append(
                    {
                        # 'rdfs': triplets,
                        'triples': s_tripleset,
                        'target': template,
                        'target_txt': text,
                        'ner2ent': ner2ent,
                    })
        if verbose and self.cnt_dirty_data: show_var(["self.cnt_dirty_data"])
        if verbose and self.cnt_corefs: show_var(["self.cnt_corefs"])

    @staticmethod
    def _triples_from_obj(obj, t_name):
        def _triples_fix(triplets):
            if not isinstance(triplets, list):
                return [triplets]
            else:
                return map(lambda t: t, triplets)

        if not isinstance(obj, list):
            if obj is not None:
                if t_name in obj:
                    return _triples_fix(obj[t_name])
            return []
        else:
            return [_triples_fix(o[t_name]) for o in obj]

    def extract_sentences(self, lex):
        sentences = lex
        if not isinstance(sentences, list): sentences = [sentences]

        for s in sentences:
            if s['@comment'] == 'bad': continue

            template = s['template']
            text = s['text']
            tag2ent = dict([(r['@tag'], r['@entity']) for r in
                            self._triples_from_obj(s['references'],
                                                   'reference')])
            s_tripleset_raw = [[tuple(map(str.strip, r.split("|")))
                                for r in
                                self._triples_from_obj(s_triples, 'striple')]
                               for s_triples in
                               self._triples_from_obj(s["sortedtripleset"],
                                                      'sentence') if
                               s_triples]
            fixed = self.fix_document(s_tripleset_raw, template, text, tag2ent)
            if fixed is None: continue
            s_tripleset, template, text, tag2ent = fixed

            if len(s_tripleset) == 1:
                template = [template]
                text = [text]
            else:
                template = self.nlp.sent_tokenize(template)
                text = self.nlp.sent_tokenize(text)
                text = fix_tokenize(text)

            if len({len(template), len(text), len(s_tripleset)}) != 1:
                # import pdb;
                # pdb.set_trace()
                self.cnt_dirty_data += 1
                continue

            for s_t, tex, tem in zip(s_tripleset, text, template):

                new_s_t, tem, uniq_tag2ent = \
                    self.fix_sentence(s_t, tem, tag2ent)
                if not (new_s_t and tem and tex and uniq_tag2ent):
                    self.cnt_corefs += 1
                    # import pdb;pdb.set_trace()
                    continue

                yield new_s_t, tex, tem, uniq_tag2ent

    def fix_document(self, s_tripleset_raw, template, text, tag2ent):
        # check template
        template = ' '.join(
            [fix_template_word[word] if word in fix_template_word else word
             for word in template.split()]) \
            if template else template

        # tokenization
        text = self.nlp.word_tokenize(text)
        template = self.nlp.word_tokenize(template)

        # clean s_tripleset
        s_tripleset = [s for s in s_tripleset_raw if s]
        self.cnt_dirty_data += len(s_tripleset_raw) - len(s_tripleset)

        if (not tag2ent) or (not s_tripleset):
            self.cnt_dirty_data += not tag2ent
            return None

        # fix this case "same entity has different ners BRIDGE-1 PATIENT-1"
        ent2tags = defaultdict(list)
        for tag, ent in tag2ent.items(): ent2tags[ent] += [tag]
        tag2uniq_tag = {}
        for ent, tags in ent2tags.items():
            for tag in tags:
                tag2uniq_tag[tag] = tags[0]
        uniq_tag2ent = {tag: ent for tag, ent in tag2ent.items()
                        if tag in tag2uniq_tag.values()}
        for tag, uniq_tag in tag2uniq_tag.items():
            template = template.replace(tag, uniq_tag)

        assert uniq_tag2ent
        ent2uniq_tag = {v: k for k, v in uniq_tag2ent.items()}
        assert len(ent2uniq_tag) == len(uniq_tag2ent)

        # clean out extra quotes around entity names
        uniq_tag2ent = {k: v.strip('\"') for k, v in uniq_tag2ent.items()}
        try:
            s_tripleset = [[(subj.strip('\"'), predi, obj.strip('\"'))
                            for subj, predi, obj in s_triples]
                           for s_triples in s_tripleset]
        except:
            import pdb;
            pdb.set_trace()

        # replaces '-' with '_' only in entity types
        tags = set(uniq_tag2ent.keys())
        for tag in tags: template = template.replace(tag, tag.replace('-', '_'))
        template = template.replace('BRIDGE-', 'BRIDGE_')
        template = template.replace('AGENT-', 'AGENT_')
        template = template.replace('PATIENT-', 'PATIENT_')
        uniq_tag2ent = {k.replace('-', '_'): v for k, v in uniq_tag2ent.items()}

        return s_tripleset, template, text, uniq_tag2ent

    def fix_sentence(self, s_tripleset, template, tag2ent):
        ent2tags = {v: k for k, v in tag2ent.items()}

        # s_tripleset must meet "head && tail are in template && tag2ent"
        bad_triples = set()
        for triple_ix, triple in enumerate(s_tripleset):
            for ent in [triple[0], triple[-1]]:
                if ent in ent2tags:
                    if ent2tags[ent] not in template:
                        bad_triples.add(triple_ix)
                        continue
                else:
                    bad_triples.add(triple_ix)
                    continue
        s_tripleset = [triple for triple_ix, triple in enumerate(s_tripleset) if
                       triple_ix not in bad_triples]

        # tag2ent are entities only in triple_entities
        triple_entities = set(flatten_list(
            [(triple[0], triple[-1]) for triple in s_tripleset]))
        tag2tri_ent = {k: v for k, v in tag2ent.items() if v in triple_entities}

        # templates only have triple_entities
        for tag, ent in tag2ent.items():
            if ent not in triple_entities:
                ent = ent.replace('_', ' ')
                template = template.replace(tag, ent)

        if {word for word in template.split()
            if 'AGENT' in word or 'BRIDGE' in word or 'PATIENT' in word} \
                != set(tag2tri_ent.keys()):
            self.cnt_corefs += 1
        assert set(tag2tri_ent.values()) == triple_entities

        '''
        TODO: 
        Erroraneous case:
        train.csv:7123:"Ayam penyet	mainIngredients	Squeezed"" or ""smashed"" fried chicken served with sambal",PATIENT_2 is PATIENT_3 .,"Fried chicken is Squeezed"" or ""smashed"" fried chicken served with sambal .",The chicken is smashed and served hot with sambal .,"Ayam penyet	Fried chicken	Squeezed"" or ""smashed"" fried chicken served with sambal",AGENT_1 PATIENT_2 PATIENT_3,ROOT	mainIngredients	mainIngredients_inv,mainIngredients,"[0, 2]","[2, 2, 8]","{""AGENT_1"": ""Ayam penyet"", ""PATIENT_2"": ""Fried chicken"", ""PATIENT_3"": ""Squeezed\"" or \""smashed\"" fried chicken served with sambal""}","[[0, 4], [4, 2], [2, 5], [5, 0]]","Ayam penyet <ENT_SEP> Fried chicken <ENT_SEP> Squeezed"" or ""smashed"" fried chicken served with sambal <ENT_REL_SEP> mainIngredients <REL_TRP_SEP> 0 2 0","Ayam penyet	mainIngredients	Squeezed"" or ""smashed"" fried chicken served with sambal <ENT_TGT_SEP> PATIENT_2 is PATIENT_3 . <TGT_TXT_SEP> The chicken is smashed and served hot with sambal ."
        train.csv:7359:Bakewell tart	ingredient	Frangipane,AGENT_1 contains PATIENT_3 .,Bakewell pudding contains Frangipane .,It contains frangipane .,Bakewell pudding	Bakewell tart	Frangipane,AGENT_1 BRIDGE_2 PATIENT_3,ROOT	ingredient	ingredient_inv,ingredient,"[1, 2]","[2, 2, 1]","{""AGENT_1"": ""Bakewell pudding"", ""BRIDGE_2"": ""Bakewell tart"", ""PATIENT_3"": ""Frangipane""}","[[1, 4], [4, 2], [2, 5], [5, 1]]",Bakewell pudding <ENT_SEP> Bakewell tart <ENT_SEP> Frangipane <ENT_REL_SEP> ingredient <REL_TRP_SEP> 1 2 0,Bakewell tart	ingredient	Frangipane <ENT_TGT_SEP> AGENT_1 contains PATIENT_3 . <TGT_TXT_SEP> It contains frangipane .
        {
            "sent": "demarce short stories in the the grantville gazettes precede eric flint novels .",
            "graph": [
                {
                    "truth": "precededBy",
                    "pred": "precededBy",
                    "ent0_ent1": "1634: the bavarian crisis ENT0_END demarce short stories in the the grantville gazettes"
                },
                {
                    "truth": "<unk>",
                    "pred": "author",
                    "ent0_ent1": "1634: the bavarian crisis ENT0_END eric flint"
                }
            ]
        }
        '''
        return s_tripleset, template, tag2tri_ent


class WebNLGDataReader(DataReader):
    def __init__(self, set: DataSetType):
        self.data_set_type = set.value
        files = self.recurse_files(
            path.join(path.dirname(path.realpath(__file__)), "raw", set.value))
        data = flatten_list([RDFFileReader(f).data for f in files])

        super().__init__(data, misspelling=misspelling,
                         rephrase=(rephrase, rephrase_if_must))

    def recurse_files(self, folder):
        if isdir(folder):
            return flatten_list(
                [self.recurse_files(folder + '/' + f) for f in listdir(folder)
                 if not f.startswith('.')])
        return [folder]

    def save(self):
        writeout = json.dumps(self.data, indent=4)
        data_set_type = 'valid' if self.data_set_type == 'dev' else self.data_set_type
        save_f = path.join(path.dirname(path.realpath(__file__)),
                           data_set_type + '.json')

        fwrite(writeout, save_f)
        print('[Info] Saved {} data into {}'.format(len(self.data), save_f))


def download():
    cmd = 'rm -rf data/webnlg/raw 2>/dev/null \n' \
          'git clone https://github.com/zhijing-jin/webnlg.git data_webnlg\n' \
          'cd data_webnlg; git checkout e978c3e; cd .. \n' \
          'cp -a data_webnlg/data/v1.5/en/ data/webnlg/raw\n' \
          'rm -rf data_webnlg\n'
    print('[Info] Downloading enriched WebNLG data...')
    shell(cmd)

def cp_data():
    cmd = 'cp data/webnlg/train.json ./ \n' \
          'cp data/webnlg/valid.json ./ \n' \
          'cp data/webnlg/test.json ./ \n'
    shell(cmd)

def main():
    download()

    for typ in DataSetType:
        data_reader = WebNLGDataReader(typ)
        data_reader.save()

    cp_data()

if __name__ == "__main__":
    main()

