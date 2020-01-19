# OpenJREE: Joint Relations and Entities Extraction

# Requirement

* python 3.7
* pytorch 1.10

# Models

## Multi-Head-Selection

[paper](https://arxiv.org/abs/1804.07847)

[official tensorflow version](https://github.com/bekou/multihead_joint_entity_relation_extraction)

## Multi-Head-Seq2Seq

## CopyMTL



# Dataset


Download the dataset from [Google Drive](https://drive.google.com/open?id=1NCwIc9-lMkKt5PxapnQy3sdRUnZiooq0)

```bash
pip install -r requirements.txt
```


```bash
mkdir raw_data
cd raw_data
unzip ../raw_data_joint.zip
```


## Chinese IE
Competition: Chinese Information Extraction Competition [link](http://lic2019.ccf.org.cn/kg)

official baseline [link](https://github.com/baidu/information-extraction/issues)

[SAOKE](https://arxiv.org/abs/1904.12535)

[official download](https://ai.baidu.com/broad/introduction?dataset=dureader)

[ai.baidu.comSAOKE2018](https://ai.baidu.com/broad/download?dataset=saoke)

Statistics:

competition

|  | avg sent length | avg triplet num | sent num |
| ------ | ------ | ------ | ------ |
| train | 47.41 | 2.06 | 157623 |
| dev   | 47.46 | 2.05 | 19628  |
| saoke | ?     | ?    | 46930  |


official data

|  SKE Dataset | avg sent length | avg triplet num | sent num |
| --------- | ------- | ------- | ------ | ----- |
| sentence  | 214,739 | 173,108 | 21,639 |19,992 |
| instance   | 47.46 | 2.05 | 19628  |

train

sentence num 155228
avg sentence length 47.062173
avg triplet num 1.878778
Counter({1: 72304, 2: 52066, 3: 16994, 4: 8032, 5: 3569, 6: 1904, 7: 202, 8: 75, 9: 41, 10: 19, 11: 14, 12: 7, 13: 1})

dev

sentence num 19365
avg sentence length 47.141492
avg triplet num 1.889285
Counter({1: 8917, 2: 6565, 3: 2107, 4: 1033, 5: 444, 6: 258, 7: 24, 8: 9, 9: 4, 10: 2, 11: 2})

**Unzip \*.json into ./raw_data/chinese/**

# Run
```shell
python main.py --mode preprocessing
python main.py --mode train
python main.py --mode evaluation

```
# Result

Training speed: 10min/epoch

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
|Ours (dev) | 0.7443 | 0.6960 | 0.7194 |
| Winner (test) | 0.8975 |0.8886 | 0.893 |


# PRs welcome

Current status
* No hyperparameter tuning
* No pretrained embedding
* No bert embedding
* No word-char embedding

Need more datasets and compared models.
