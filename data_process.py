#!usr/bin/env python
#-*- coding:utf-8 -*-

import random
import pandas as pd
from data_loader import load_dataset, save_data
from sklearn.model_selection import train_test_split


def gen_char_dict(dataset):
    char_dict = {}
    for line in dataset:
        line = line.strip()
        for char in line:
            if len(char) != 0:
                char_dict[char] = char_dict.get(char, 0) + 1
    return char_dict


def random_word(sentence, char_dict):
    tokens = [x for x in sentence]
    out = []
    for i, token in enumerate(sentence):
        if not token.isalpha():
            out.append(str(0))
            continue
        # randomly pickup an index of the item
        prob = random.random()
        if prob < 0.20:
            # random 5%
            if prob < 0.05:
                candiation = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation if x[0] != token]
                if len(candiation) == 0:
                    out.append(str(0))
                    continue
                tokens[i] = random.choice(candiation)
                out.append(str(1))
            # add 5%
            # elif prob < 0.10:
            #     if i == (len(tokens) - 1):
            #         candiation = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
            #         candiation = candiation[:int(len(candiation)/2+0.5)]
            #         candiation = [x[0] for x in candiation]
            #         tokens.insert(i+1, random.choice(candiation))
            #         out.append(str(0))
            #         out.append(str(1))
            #     else:
            #         continue
            # delete 5%
            # elif prob < 0.15:
            #     tokens[i] = ''
            # # transpose 5%
            else:
                if i == (len(tokens) - 1):
                    continue
                else:
                    tokens[i] = tokens[i+1]
                out.append(str(1))
        # original 80%
        else:
            out.append(str(0))
    return ''.join(tokens), ' '.join(out)


def random_dataset(dataset, char_dict):
    text = []
    out = []
    for ids, line in enumerate(dataset):
        line, label = random_word(line, char_dict)
        text.append(line)
        out.append(label)
    return text, out


if __name__ == '__main__':
    dataset = load_dataset('dataset/sentences.txt')
    char_dict = gen_char_dict(dataset)
    process_dataset, process_label = random_dataset(dataset, char_dict)
    save_data(process_dataset, 'dataset/sentences_noisy.txt')
    df = pd.DataFrame(columns=['original','noisy','label'])
    df['original'] = dataset
    df['noisy'] = process_dataset
    df['label'] = process_label
    # remove nan rows
    df.dropna(inplace=True)
    # df.to_csv('dataset/processed_data.csv', index=False)

    # df = pd.read_csv('dataset/processed_words.csv')
    dataset = df[['original','noisy','label']].values
    train, test = train_test_split(dataset, test_size=0.1)
    df = pd.DataFrame(columns=['original','noisy','label'], data=train)
    df.to_csv('dataset/processed_train.csv', index=False)
    df = pd.DataFrame(columns=['original','noisy','label'], data=test)
    df.to_csv('dataset/processed_test.csv', index=False)


