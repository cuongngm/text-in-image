import collections
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
STRING_MAX_LEN = 80
VOCABULARY_FILE_NAME = '../keysVN.txt'


class ResizeWeight(object):

    def __init__(self, size, interpolation=Image.BILINEAR, gray_format=True):
        self.w, self.h = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.gray_format = gray_format

    def __call__(self, img):
        img_w, img_h = img.size

        if self.gray_format:
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:self.h, 0] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:new_w, 0] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[:, :, 0] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w
        else:  # RGB format
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:self.h, :] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:new_w, :] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[:, :, :] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w


class ConvertLabelToMASTER(object):
    def __init__(self, vocab_file, max_length=-1, ignore_over=False):
        with open(vocab_file, 'r') as file:
            classes = file.read()
            classes = classes.strip()
            cls_list = list(classes)

        self.alphabet = cls_list
        self.dict = {}
        self.dict['<EOS>'] = 1
        self.dict['<SOS>'] = 2
        self.dict['<PAD>'] = 0
        self.dict['<UNK>'] = 3

        self.EOS = self.dict['<EOS>']
        self.SOS = self.dict['<SOS>']
        self.PAD = self.dict['<PAD>']
        self.UNK = self.dict['<UNK>']

        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 4
        self.inverse_dict = {v: k for k, v in self.dict.items()}
        self.nclass = len(self.alphabet) + 4
        self.max_length = max_length
        self.ignore_over = ignore_over

    def encode(self, text):
        # return: torch.LongTensor(max_length x batch_size)
        if isinstance(text, str):
            text = [self.dict[item] if item in self.alphabet else self.UNK for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
            if self.max_length == -1:
                local_max_length = max([len(x) for x in text])
                self.ignore_over = True
            else:
                local_max_length = self.max_length
            nb = len(text)
            targets = torch.zeros(nb, local_max_length + 2)
            targets[:, :] = self.PAD
            for i in range(nb):
                if not self.ignore_over:
                    if len(text[i]) > local_max_length:
                        raise RuntimeError('Text is larger than {}:{}'.format(local_max_length, text[i]))
                targets[i][0] = self.SOS
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = self.EOS
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        # decode back to string
        if isinstance(t, torch.Tensor):
            texts = self.inverse_dict[t.item()]
        else:
            texts = self.inverse_dict[t]
        return texts


def subsequent_mask(tgt, padding_symbol):
    """
    tag: (bs, seq_len)
    """
    trg_pad_mask = (tgt != padding_symbol).unsqueeze(1).unsqueeze(3) # (bs, 1, seq_len, 1)
    tgt_len = tgt.size(1) # seq_len,
    trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8)).to(tgt.device) # (seq_len, seq_len)
    tgt_mask = trg_pad_mask & trg_sub_mask.bool()
    return tgt_mask # (bs, 1, seq_len, seq_len)


def greedy_decode(model, input, max_len, start_symbol, padding_symbol=None, device='cpu', padding=False):
    """
    output predicted transcript
    :param model:
    :param input:
    :param max_len:
    :param start_symbol:
    :param padding_symbol:
    :param device:
    :param padding: if padding is True, max_len will be used. if paddding is False and max_len == -1, max_len will
    be set to 100, otherwise max_len will be used.
    :return:
    """
    B = input.size(0)
    memory = model.encode(input, None)
    if padding:
        if padding_symbol is None:
            raise RuntimeError('Padding Symbol cannot be None.')

        assert max_len > 0

        ys = torch.ones((B, max_len + 2), dtype=torch.long).fill_(padding_symbol).to(device)
        ys[:, 0] = start_symbol
    else:
        if max_len == -1:
            max_len = 100
        ys = torch.ones((B, 1), dtype=torch.long).fill_(start_symbol).to(device)

        # decode with max_len + 1 time step, (include eos）
    for i in range(max_len + 1):
        out = model.decode(memory, None, ys, subsequent_mask(ys, padding_symbol).to(device))
        out = model.generator(out)
        prob = F.softmax(out, dim=-1)
        _, next_word = torch.max(prob, dim=-1)

        if padding:
            ys[:, i + 1] = next_word[:, i]
        else:
            ys = torch.cat([ys, next_word[:, -1].unsqueeze(-1)], dim=1)

    return ys


def greedy_decode_with_probability(model, input, max_len, start_symbol, padding_symbol=None,
                                   device='cpu', padding=False):
    B = input.size(0)
    memory = model.encode(input, None)

    if padding:
        if padding_symbol is None:
            raise RuntimeError('Padding Symbol cannot be None.')

        assert max_len > 0

        ys = torch.ones((B, max_len + 2), dtype=torch.long).fill_(padding_symbol).to(device)
        probs = torch.ones((B, max_len + 2), dtype=torch.float).to(device)
        ys[:, 0] = start_symbol
        # probs[:, 0] = 1.0
    else:
        if max_len == -1:
            max_len = 100
        ys = torch.ones((B, 1), dtype=torch.long).fill_(start_symbol).to(device)
        probs = torch.ones((B, 1), dtype=torch.float).to(device)
    # decode with max_len + 1 time step, (include eos）
    for i in range(max_len + 1):
        out = model.decode(memory, None, ys, subsequent_mask(ys, padding_symbol).to(device))
        out = model.generator(out)

        prob = F.softmax(out, dim=-1)
        max_probs, next_word = torch.max(prob, dim=-1)  # (bs, t)

        if padding:
            ys[:, i + 1] = next_word[:, i]
            probs[:, i + 1] = max_probs[:, i]
        else:
            ys = torch.cat([ys, next_word[:, -1].unsqueeze(-1)], dim=1)
            probs = torch.cat([probs, max_probs[:, -1].unsqueeze(-1)], dim=1)

    return ys, probs
