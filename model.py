from transformers import RobertaModel, XLMRobertaModel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.init as init


################################################
class PhobertForQuestionAnsweringAVPool(nn.Module):
    def __init__(self, model_path, config):
        super(PhobertForQuestionAnsweringAVPool, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.phobert = RobertaModel.from_pretrained(model_path, config= config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.has_ans.weight.data)
        self.has_ans.bias.data.uniform_(0, 0)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(self.dropout(first_word))

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class XLMRobertaForQuestionAnsweringAVPool(nn.Module):
    def __init__(self, model_path, config):
        super(XLMRobertaForQuestionAnsweringAVPool, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.phobert = XLMRobertaModel.from_pretrained(model_path, config= config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.qa_outputs.weight.data)
        self.qa_outputs.bias.data.uniform_(0, 0)
        init.xavier_uniform_(self.has_ans.weight.data)
        self.has_ans.bias.data.uniform_(0, 0)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(self.dropout(first_word))

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

####################################################
from modeling import RobertaLayer
class HSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(HSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pre_layers = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weight()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))

    def init_weight(self):
        init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.uniform_(0, 0)

    def forward(self, layers, attention_mask):
        logitses = []
        output = torch.zeros_like(layers[0])

        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)[0]
            logits = self.classifier(output)
            logitses.append(logits)

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return avg_logits

class XLM_MIXLAYER_single(nn.Module):
    def __init__(self, model_path, config, count= 3, mix_type= "HSUM"):
        super(XLM_MIXLAYER_single, self).__init__()
        self.xlmroberta = XLMRobertaModel.from_pretrained(model_path, config=config)
        if mix_type.upper() == "HSUM":
            self.mixlayer = HSUM(count, config, 2)
        # elif mix_type.upper() == "PSUM":
        #     self.mixlayer = PSUM(count, config_phobert, num_classes)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):
    
        outputs = self.xlmroberta(input_ids= input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states= True)
        layers = outputs[2]
        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        logits = self.mixlayer(layers, extend_attention_mask)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output