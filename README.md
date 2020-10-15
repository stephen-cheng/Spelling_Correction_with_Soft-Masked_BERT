## Soft-Masked Bert Model

- Reference Paper: [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/pdf/2005.07421.pdf)

- Dataset:  The data that we will use for this project will be 20 popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads).


## Prerequired packages

```
pip install -r requirements.txt
```

## Parameters:

The length of each sentence is between 4 and 200. So,

- max_len = 200
- min_len = 4


## How to run?

- Prepare Data: python data_prepare.py

- Process Data: python data_process.py

- Train Models: python train.py