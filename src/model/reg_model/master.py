import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_function import create_module


class Generator(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for m_module_index, m_module in enumerate(self):
            if m_module_index == 0:
                m_input = m_module(*inputs)
            else:
                m_input = m_module(m_input)
        return m_input


class MASTER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.with_encoder = config['model_arch']['common']['with_encoder']

        for m_parameter in self.parameters():
            if m_parameter.dim() > 1:
                nn.init.xavier_uniform_(m_parameter)

        self.conv_embedding_gc = create_module(
            config['functional']['conv_embedding_gc'])(config)
        self.encoder = create_module(config['functional']['encoder'])(config['model_arch']['common']['with_encoder'],
                                                                      config['model_arch']['common']['nhead'],
                                                                      config['model_arch']['common']['d_model'],
                                                                      config['model_arch']['encoder']['num_layer'],
                                                                      config['model_arch']['encoder']['dropout'],
                                                                      config['model_arch']['encoder']['ff_dim'],
                                                                      config['model_arch']['encoder']['share_parameter'])
        self.encode_stage = nn.Sequential(self.conv_embedding_gc, self.encoder)

        self.decoder = create_module(config['functional']['decoder'])(config['model_arch']['common']['nhead'],
                                                                      config['model_arch']['common']['d_model'],
                                                                      config['model_arch']['decoder']['num_layer'],
                                                                      config['model_arch']['decoder']['dropout'],
                                                                      config['model_arch']['decoder']['ff_dim'],
                                                                      config['model_arch']['common']['tgt_vocab'])
        self.generator = create_module(config['functional']['generator'])(config['model_arch']['common']['d_model'],
                                                                          config['model_arch']['common']['tgt_vocab'])
        self.decode_stage = MultiInputSequential(self.decoder, self.generator)

    def eval(self):
        self.conv_embedding_gc.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.generator.eval()

    def forward(self, src, tgt):
        encode_stage_result = self.encode_stage(src)
        decode_stage_result = self.decode_stage(tgt, encode_stage_result)
        return decode_stage_result

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
