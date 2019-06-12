# OpenJREE: Joint Relations and Entities Extraction

# Requirement

* python 3.7
* pytorch 1.10

# Models

## Multi-Head-Selection

[paper](https://arxiv.org/abs/1804.07847)

[official tensorflow version](https://github.com/bekou/multihead_joint_entity_relation_extraction)

# Dataset

## Chinese IE
Chinese Information Extraction Competition [link](http://lic2019.ccf.org.cn/kg)

Statistics:

|  | avg sent length | avg triplet num | sent num |
| ------ | ------ | ------ | ------ |
| train | 47.41 | 2.06 | 157623 |
| dev | 47.46 | 2.05 | 19628 |

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
