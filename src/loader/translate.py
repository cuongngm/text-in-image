import collections
from pathlib import Path
import torch


class LabelConverter:
    def __init__(self, classes, max_length=-1, ignore_over=False):
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, list):
            cls_list = classes
        elif isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        self.alphabet = cls_list
        self.alphabet_mapper = {'<EOS>': 1, '<SOS>': 2, '<PAD>': 0, '<UNk>': 3}
        for i, item in enumerate(self.alphabet):
            self.alphabet_mapper[item] = i + 4
        self.alphabet_inverse_mapper = {v: k for k, v in self.alphabet_mapper.items()}
        self.EOS = self.alphabet_mapper['<EOS>']
        self.SOS = self.alphabet_mapper['<SOS>']
        self.PAD = self.alphabet_mapper['<PAD>']
        self.UNK = self.alphabet_mapper['<UNK>']
        self.n_class = len(self.alphabet) + 4
        self.max_length = max_length
        self.ignore_over = ignore_over

    def encode(self, text):
        if isinstance(text, str):
            text = [self.alphabet_mapper[item] if item in self.alphabet else self.UNK for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
            if self.max_length == -1:
                local_max_length = max([len(x) for x in text])
                self.ignore_over = True
            else:
                local_max_length = self.max_length
            nb = len(text)
            targets = torch.zeros(nb, (local_max_length  + 2))
            targets[:, :] = self.PAD
            for i in range(nb):
                if not self.ignore_over:
                    if len(text[i]) > local_max_length:
                        raise RuntimeError('Text is larger than {}: {}'.format(local_max_length, len(text[i])))
                targets[i][0] = self.SOS
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = self.EOS
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        if isinstance(t, torch.Tensor):
            texts = self.alphabet_inverse_mapper[t.item()]
        else:
            texts = self.alphabet_inverse_mapper[t]
        return texts


LabelTransformer = LabelConverter(Path(__file__).parent.joinpath('keysVN.txt'),
                                  max_length=100, ignore_over=False)
