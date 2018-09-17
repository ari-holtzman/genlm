import os, pickle
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path, need_data=True):
        dic_path = os.path.join(path, 'dic.pickle')
        if os.path.exists(dic_path):
            with open(dic_path, 'rb') as dic_file:
                self.dictionary = pickle.load(dic_file)
        else:
            self.dictionary = Dictionary()
        if need_data:
            if os.path.exists(os.path.join(path, 'train.pt')):
                self.train = torch.load(os.path.join(path, 'train.pt'), map_location='cpu')
            else:
                self.train = self.tokenize(os.path.join(path, 'train.txt'))
                torch.save(self.train, os.path.join(path, 'train.pt'))
            if os.path.exists(os.path.join(path, 'valid.pt')):
                self.valid = torch.load(os.path.join(path, 'valid.pt'), map_location='cpu')
            else: 
                self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
                torch.save(self.valid, os.path.join(path, 'valid.pt'))
            if os.path.exists(os.path.join(path, 'test.pt')):
                self.test = torch.load(os.path.join(path, 'test.pt'), map_location='cpu')
            else:
                self.test = self.tokenize(os.path.join(path, 'test.txt'))
                torch.save(self.test, os.path.join(path, 'test.pt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
