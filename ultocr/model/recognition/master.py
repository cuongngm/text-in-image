import numpy as np
import torch
import torch.nn as nn
from ultocr.utils.utils_function import create_module


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


def predict(memory, src, decode_stage, max_length, sos_symbol, eos_symbol, padding_symbol):
    batch_size = src.size()
    device = src.device
    to_return_label = torch.ones((batch_size, max_length + 2), dtype=torch.long).to(device) * padding_symbol
    prob = torch.ones((batch_size, max_length + 2), dtype=torch.float32).to(device)
    to_return_label[:, 0] = sos_symbol
    for i in range(max_length + 1):
        m_label = decode_stage(to_return_label, memory)
        m_prob = torch.softmax(m_label, dim=-1)
        m_max_probs, m_next_word = torch.max(m_prob, dim=-1)
        to_return_label[:, i + 1] = m_next_word[:, i]
        prob[:, i + 1] = m_max_probs[:, i]
    eos_position_y, eos_position_x = torch.nonzero(to_return_label == eos_symbol, as_tuple=True)
    if len(eos_position_y) > 0:
        eos_position_y_index = eos_position_y[0]
        for m_position_y, m_position_x in zip(eos_position_y, eos_position_x):
            if eos_position_y_index == m_position_y:
                to_return_label[m_position_y, m_position_x + 1:] = padding_symbol
                prob[m_position_y, m_position_x + 1:] = 1
                eos_position_y_index += 1
    return to_return_label, prob


def greedy_decode_with_probability(model, input_tensor, max_seq_len, sos_symbol_idx, eos_symbol_idx,
                                   pad_symbol_idx=None, result_device='cpu', is_padding=False):
    memory = model.encode_stage(input_tensor)
    predicted_label, predicted_label_prob = predict(memory, input_tensor, model.decode_stage,
                                                    max_seq_len, sos_symbol_idx, eos_symbol_idx,
                                                    pad_symbol_idx)
    return predicted_label, predicted_label_prob
