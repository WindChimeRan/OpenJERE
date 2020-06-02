# OpenJREE: Joint Relations and Entities Extraction

# Requirement

* python 3.7
* pytorch 1.5

# Models

* Multi-Head-Selection [paper](https://arxiv.org/abs/1804.07847)
* CopyMTL [paper](https://arxiv.org/pdf/1911.10438.pdf)
* WDec [paper](https://128.84.21.199/pdf/1911.09886.pdf)

* Seq2UMTree

# Run


Download the DuIE dataset from [official website](https://ai.baidu.com/broad/introduction?dataset=dureader)

Unzip \*.json into ./raw_data/chinese/

For NYT, see raw_data/nyt/README.md


```bash
pip install -r requirements.txt
```


```bash
cd raw_data
unzip ../raw_data_joint.zip
```

<!-- Then use the script to download enriched webnlg directly:

```bash
cd raw_data/EWebNLG
python data/webnlg/reader.py
``` -->

Then run data_split for both datasets:
```bash
python data_split.py
```

```bash
bash train_all.sh
```

seperate steps:

```shell
python main.py --mode preprocessing --exp chinese_seq2umt_ops
python main.py --mode train --exp chinese_seq2umt_ops
python main.py --mode evaluation --exp chinese_seq2umt_ops
```

```shell
python main.py --mode preprocessing --exp nyt_seq2umt_ops
python main.py --mode train --exp nyt_seq2umt_ops
python main.py --mode evaluation --exp nyt_seq2umt_ops
```

```shell
python main.py --mode preprocessing --exp nyt_wdec
python main.py --mode train --exp nyt_wdec
python main.py --mode evaluation --exp nyt_wdec
```

<!-- ## EWebNLG

[code](https://github.com/zhijing-jin/WebNLG_Reader)
[paper](https://www.aclweb.org/anthology/W18-6521.pdf)

 -->


