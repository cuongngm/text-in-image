import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from ultocr.utils.download import download_weights
from ultocr.utils.utils_function import create_module
from ultocr.loader.recognition.translate import LabelConverter
from ultocr.model.common.transformer import EncoderLayer, DecoderLayer, Encoder, Decoder, Embeddings


class Generator(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, *input):
        x = input[0]
        return self.fc(x)


class MASTER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_file = download_weights('1Lo9L_k63M7vpiR10zii5nzL4GGUSuntM')
        self.convert = LabelConverter(classes=vocab_file, max_length=100, ignore_over=False)
        tgt_vocab = self.convert.n_class
        self.with_encoder = config['model']['common']['with_encoder']
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        conv_embedding_gc = create_module(config['functional']['conv_embedding_gc'])(config)
        embedding = create_module(config['functional']['embedding'])(
            config['model']['common']['d_model'],
            tgt_vocab
        )
        encoder_attn = create_module(config['functional']['multi_head_attention'])(
            config['model']['common']['nhead'],
            config['model']['common']['d_model'],
            config['model']['encoder']['dropout']
        )
        encoder_ff = create_module(config['functional']['feed_forward'])(
            config['model']['common']['d_model'],
            config['model']['encoder']['ff_dim'],
            config['model']['encoder']['dropout']
        )
        encoder_position = create_module(config['functional']['position'])(
            config['model']['common']['d_model'],
            config['model']['encoder']['dropout']
        )

        decoder_attn = create_module(config['functional']['multi_head_attention'])(
            config['model']['common']['nhead'],
            config['model']['common']['d_model'],
            config['model']['decoder']['dropout']
        )
        decoder_ff = create_module(config['functional']['feed_forward'])(
            config['model']['common']['d_model'],
            config['model']['decoder']['ff_dim'],
            config['model']['decoder']['dropout']
        )
        decoder_position = create_module(config['functional']['position'])(
            config['model']['common']['d_model'],
            config['model']['decoder']['dropout']
        )

        if self.with_encoder:
            encoder = Encoder(EncoderLayer(size=config['model']['common']['d_model'],
                                           self_attn=deepcopy(encoder_attn),
                                           feed_forward=deepcopy(encoder_ff),
                                           dropout=config['model']['encoder']['dropout']),
                              config['model']['encoder']['num_layer'])
        else:
            encoder = None
        decoder = Decoder(DecoderLayer(size=config['model']['common']['d_model'],
                                       self_attn=deepcopy(decoder_attn),
                                       src_attn=deepcopy(decoder_attn),
                                       feed_forward=deepcopy(decoder_ff),
                                       dropout=config['model']['decoder']['dropout']),
                          config['model']['decoder']['num_layer'])
        self.generator = create_module(config['functional']['generator'])(config['model']['common']['d_model'],
                                                                          tgt_vocab)
        src_embed = nn.Sequential(conv_embedding_gc, deepcopy(encoder_position))
        tgt_embed = nn.Sequential(Embeddings(config['model']['common']['d_model'], tgt_vocab), deepcopy(decoder_position))
        padding = self.convert.PAD
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.padding = padding
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        """
        :param src:
        :param src_mask:
        :return:
        """
        if self.with_encoder:  # cnn + encoder + decoder
            return self.encoder(self.src_embed(src), src_mask)
        else:  # cnn + decoder
            return self.src_embed(src)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        :param memory: output from encoder
        :param src_mask:
        :param tgt: raw target input (label of text squence)
        :param tgt_mask: [b, h, len_seq, len_seq]
        :return:
        """

        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def make_mask(self, src, tgt):
        """
        :param src: [b, c, h, len_src]
        :param tgt: [b, l_tgt]
        :return:
        """

        # src_mask does not need, since the embedding generated by ConvNet is dense.
        trg_pad_mask = (tgt != self.padding).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask.bool()
        return None, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_mask(src, tgt)
        # output = self.decode(src, src_mask, tgt, tgt_mask)
        output = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return self.generator(output)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
