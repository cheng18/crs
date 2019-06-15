# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random

import numpy as np

from typing import List

import csv
import json
import os

class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    '''
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, stroke_vocab_file=None, **kwargs): # Winfred 筆畫字典
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        num_words = len(self._id_to_word)

        # Winfred 若 do_stroke， self._word_char_ids 裏除了 {word: char} 也增加 {中文char: stroke}
        self._word_char_ids = np.zeros([num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        # Add by Winfred
        print('==========================do_?===========================')
        print('stroke_vocab_file', stroke_vocab_file)
        if stroke_vocab_file is not None:
            self.stroke_vocab = load_stroke_vocab(stroke_vocab_file) # Winfred
            self.do_stroke = True
            print('==========================do_stroke===========================')
        else:
            self.do_stroke = False
        # End

        for i, word in enumerate(self._id_to_word):
            # Add by Winfred
            if self.do_stroke and len(word) == 1 and is_chinese_char(ord(word)):
                char = word
                self._word_char_ids[i] = self._convert_char_to_stroke_ids(char)
            # End
            else:
                self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK> # ??


    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    # Add by Winfred
    def _convert_char_to_stroke_ids(self, char):
        stroke_ids = np.zeros([self.max_word_length], dtype=np.int32)
        stroke_ids[:] = self.pad_char
        if char in self.stroke_vocab:
            stroke = self.stroke_vocab[char][:(self.max_word_length-2)] #??
            stroke_ids[0] = self.bow_char
            for k, stroke_id in enumerate(stroke, start=1):
                stroke_ids[k] = stroke_id + 260
            stroke_ids[len(stroke) + 1] = self.eow_char

        return stroke_ids
    # End

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            # Add by Winfred
            if self.do_stroke and len(word) == 1 and is_chinese_char(ord(word)):
                char = word
                return self._convert_char_to_stroke_ids(char)
            # End
            else:
                return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


class Batcher(object):
    ''' 
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file: str, max_token_length: int,
                 max_seq_length=None, stroke_vocab_file=None): # Add by Winfred
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        max_length = 句子最大長度
        '''
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length,
            stroke_vocab_file=stroke_vocab_file # Add by Winfred
        )
        self._max_token_length = max_token_length
        self._max_seq_length = max_seq_length # Add by Winred

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        if self._max_seq_length is None: # Add by Winfred
            max_length = max(len(sentence) for sentence in sentences) + 2 # ？ Why + 2, line 288
        else:
            max_length = self._max_seq_length + 2 # Add by Winfred

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            if self._max_seq_length and len(sent) > self._max_seq_length:   # Add by Winfred
                sent = sent[:self._max_seq_length] # Add by Winfred
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object):
    ''' 
    Batch sentences of tokenized text into token id matrices.
    '''
    def __init__(self, lm_vocab_file: str,
                 max_seq_length=None): # Add by Winfred
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = Vocabulary(lm_vocab_file)
        self._max_seq_length = max_seq_length # Add by Winred

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        if self._max_seq_length is None: # Add by Winfred
            max_length = max(len(sentence) for sentence in sentences) + 2
        else:
            max_length = self._max_seq_length + 2 # Add by Winfred

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            if self._max_seq_length and len(sent) > self._max_seq_length:   # Add by Winfred
                sent = sent[:self._max_seq_length] # Add by Winfred
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        return X_ids


##### for training
def _get_batch(generator, batch_size, num_steps, max_word_length):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_word_length is not None:
            char_inputs = np.zeros([batch_size, num_steps, max_word_length],
                                np.int32)
        else:
            char_inputs = None
        targets = np.zeros([batch_size, num_steps], np.int32)

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                                                                    :how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]

                cur_pos = next_pos

                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

        if no_more_data:
            # There is no more data.  Note: this will not return data
            # for the incomplete batch
            break

        X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                 'next_token_id': targets}

        yield X

class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.  Each file contains one sentence
        per line.  Each sentence is pre-tokenized and white space joined.
    """
    def __init__(self, filepattern, vocab, reverse=False, test=False,
                 shuffle_on_load=False,
                 do_record=False, records_path=None, vocab_file=None): # Add by Winfred
        '''
        filepattern = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        '''
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        # Add by Winfred
        self.recorder = None
        if do_record:
            assert records_path is not None
            assert vocab_file is not None
            self.recorder = Recorder(records_path, vocab_file)
        # End

        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()
        print('Loading ok') # Add by Winfred

        sentences_raw = [tokenize_chinese_chars(sentence) for sentence in sentences_raw] # Winfred 中文加空格
        
        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)
            print('shuffled') # Add by Winfred
        print('shuffle ok') # Add by Winfred

        # Add by Winfred
        if self.recorder is not None:
            self.recorder.save_sentences(sentences)
            print('recorder save_sentences') # Add by Winfred
        print('recorder ok') # Add by Winfred
        # End

        ids = [self.vocab.encode(sentence, self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self.vocab.encode_chars(sentence, self._reverse)
                     for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        print('Loaded %d sentences.' % len(ids))
        print('Finished loading')
        return list(zip(ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            # Add by Winfred
            if self.recorder is not None:
                self.recorder.i_add_one()
            # End
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(self.get_sentence(), batch_size, num_steps,
                           self.max_word_length):

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab

class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False,
                 do_record=False, records_path=None, vocab_file=None): # Add by Winfred
        '''
        bidirectional version of LMDataset
        '''
        self._data_forward = LMDataset(
            filepattern, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load,
            do_record=do_record, records_path=records_path, vocab_file=vocab_file) # Add by Winfred
        self._data_reverse = LMDataset(
            filepattern, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        for X, Xr in zip(
            _get_batch(self._data_forward.get_sentence(), batch_size,
                      num_steps, max_word_length),
            _get_batch(self._data_reverse.get_sentence(), batch_size,
                      num_steps, max_word_length)
            ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X

    # Add by Winfred
    def save_recorder(self):
        if self._data_forward.recorder is not None:
            self._data_forward.recorder.run_record()
    # End


class InvalidNumberOfCharacters(Exception):
    pass


# add by winfred
def load_stroke_vocab(stroke_vocab_file):
  """Loads a stroke vocabulary file into a dictionary."""
  with open(stroke_vocab_file, newline="", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    header = next(reader)
    char_index = header.index("汉字")
    stroke_index = header.index("笔顺")

    stroke_vocab = {}
    for row in reader:
      stroke_vocab[row[char_index]] = list(map(int, row[stroke_index]))

  return stroke_vocab
# end

# Winfred 判斷是不是中文
def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char) 
    return "".join(output)


class Recorder(object):
    """
    記錄 pre-training dataset 與 vocab 的關係，瞭解哪些字詞沒預訓練到。
    輸出 record.json = {"vocab_words": counts, 
                       "UNK_list": {"UNKs": counts}}
    """
    def __init__(self, records_path, vocab_file):
        self.records_path = records_path
        self.vocab_file = vocab_file
        self.records_file = os.path.join(self.records_path, "records.json")
        self.sentences_temp_file = os.path.join(self.records_path, "sentences_temp")
        self.i = 0
        
        print("Recorder init")
    
    def _load_vocab_to_records(self):
        records = {"UNK_list": {}}
        with open(self.vocab_file) as lines:
            for line in lines:
                line = line.strip()
                records[line] = 0
        return records

    def _record_sentences(self, records, sentences):
        for sentence in sentences[:self.i+1]:
            for token in sentence.split():
                if token in records:
                    records[token] += 1
                else:
                    records["<UNK>"] += 1
                    if token in records["UNK_list"]:
                        records["UNK_list"][token] += 1
                    else:
                        records["UNK_list"][token] = 1
        return records
    
    def _count_tokens(self, sentences):
        return sum([len(sentence.split()) for sentence in sentences])
    
    def i_add_one(self):
        self.i += 1

    def save_sentences(self, sentences):
        with open(self.sentences_temp_file, "a") as f:
            f.writelines(sentences)
        print("tokens number: %d" % self._count_tokens(sentences))
    
    def run_record(self):
        records = self._load_vocab_to_records()
        with open(self.sentences_temp_file, "r") as f:
            sentences = f.readlines()
        records = self._record_sentences(records, sentences)
        json_str = json.dumps(records, ensure_ascii=False)
        with open(self.records_file, "w", encoding='utf8') as f:
            f.write(json_str)
