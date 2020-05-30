# OpenJREE: Joint Relations and Entities Extraction

# Requirement

* python 3.7
* pytorch 1.5

# Models

a. Multi-Head-Selection [paper](https://arxiv.org/abs/1804.07847)
b. CopyMTL [paper](https://arxiv.org/pdf/1911.10438.pdf)
c. WDec [paper](https://128.84.21.199/pdf/1911.09886.pdf)


# Run


Download the DuIE dataset from [Google Drive](https://drive.google.com/open?id=1NCwIc9-lMkKt5PxapnQy3sdRUnZiooq0)

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


## Chinese IE
Competition: Chinese Information Extraction Competition [link](http://lic2019.ccf.org.cn/kg)

official baseline [link](https://github.com/baidu/information-extraction/issues)

[SAOKE](https://arxiv.org/abs/1904.12535)

[official download](https://ai.baidu.com/broad/introduction?dataset=dureader)

[ai.baidu.comSAOKE2018](https://ai.baidu.com/broad/download?dataset=saoke) -->


