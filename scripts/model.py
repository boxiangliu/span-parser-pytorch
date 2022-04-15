from unicodedata import bidirectional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from parser import Parser


class Network(nn.Module):
    def __init__(
        self,
        feature_mapper,
        word_count,
        tag_count,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        struct_out,
        label_out,
        alpha,
        beta,
        unk_param=0.8375,
        droprate=0,
        struct_spans=4,
        label_spans=3,
        feature="subtract",
        dynamic_oracle=True,
    ):
        super().__init__()
        self.word_count = word_count
        self.tag_count = tag_count
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.struct_out = struct_out
        self.label_out = label_out
        self.unk_param = unk_param
        self.fm = feature_mapper
        self.alpha = alpha
        self.beta = beta
        self.droprate = droprate
        self.feature = feature
        self.dynamic_oracle = dynamic_oracle

        self.word_embed = nn.Embedding(word_count, word_dims)
        self.tag_embed = nn.Embedding(tag_count, tag_dims)  # POS tags

        self.lstm = nn.LSTM(
            input_size=word_dims + tag_dims,
            hidden_size=lstm_units,
            num_layers=2,
            bidirectional=True,
            dropout=droprate,
        )

        self.struct_fc1 = nn.Linear(2 * struct_spans * lstm_units, hidden_units)
        self.struct_fc2 = nn.Linear(hidden_units, struct_out)

        self.label_fc1 = nn.Linear(2 * label_spans * lstm_units, hidden_units)
        self.label_fc2 = nn.Linear(hidden_units, label_out)

        self.dropout = nn.Dropout(droprate)
        self.criterion = nn.CrossEntropyLoss()

    def _evaluate_recurrent(self, word_inds, tag_inds):
        # [len, word_dims]
        word_embedded = self.word_embed(word_inds)

        # [len, tag_dims]
        tag_embedded = self.tag_embed(tag_inds)

        # [len, word_dims + tag_dims]
        embedded = torch.cat([word_embedded, tag_embedded], dim=1)
        embedded = embedded.unsqueeze(1)

        # lstm_outputs: [len, 1, lstm_units * 2]
        lstm_outputs, (hn, cn) = self.lstm(embedded)

        return lstm_outputs

    def evaluate_recurrent(self, word_inds, tag_inds, test=False):
        if test:
            self.eval()
            with torch.no_grad():
                lstm_outputs = self._evaluate_recurrent(word_inds, tag_inds)
            self.train()
        else:
            lstm_outputs = self._evaluate_recurrent(word_inds, tag_inds)
        return lstm_outputs

    def evaluate_struct(self, lstm_outputs, lefts, rights, test=False):
        fwd_out = lstm_outputs[:, :, : self.lstm_units]
        back_out = lstm_outputs[:, :, self.lstm_units :]

        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            if self.feature == "subtract":
                fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
            elif self.feature == "sum":
                # Boxiang trying sum instead of subtraction.
                fwd_span_out.append(fwd_out[left_index - 1 : right_index].sum(dim=0))
        # [N, lstm_units * struct_span]
        fwd_span_vec = torch.cat(fwd_span_out, dim=1)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            if self.feature == "subtract":
                back_span_out.append(back_out[left_index] - back_out[right_index + 1])
            elif self.feature == "sum":
                back_span_out.append(back_out[left_index : right_index + 1].sum(dim=0))
        # [N, lstm_units * struct_span]
        back_span_vec = torch.cat(back_span_out, dim=1)
        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=1)
        hidden_input = self.dropout(hidden_input)
        hidden_output = F.relu(self.struct_fc1(hidden_input))
        scores = self.struct_fc2(hidden_output)

        return scores

    def evaluate_label(self, lstm_outputs, lefts, rights, test=False):

        fwd_out = lstm_outputs[:, :, : self.lstm_units]
        back_out = lstm_outputs[:, :, self.lstm_units :]

        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            if self.feature == "subtract":
                fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
            elif self.feature == "sum":
                # Boxiang trying sum instead of subtraction.
                fwd_span_out.append(fwd_out[right_index : left_index - 1].sum(dim=0))
        # [N, lstm_units * struct_span]
        fwd_span_vec = torch.cat(fwd_span_out, dim=1)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            if self.feature == "subtract":
                back_span_out.append(back_out[left_index] - back_out[right_index + 1])
            elif self.feature == "sum":
                back_span_out.append(back_out[left_index : right_index + 1].sum(dim=0))
        # [N, lstm_units * struct_span]
        back_span_vec = torch.cat(back_span_out, dim=1)

        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=1)
        hidden_input = self.dropout(hidden_input)

        hidden_output = F.relu(self.label_fc1(hidden_input))
        scores = self.label_fc2(hidden_output)

        return scores

    def forward(self, batch):
        if self.dynamic_oracle:
            explore = [
                Parser.exploration(
                    example, self.fm, self, alpha=self.alpha, beta=self.beta
                )
                for example in batch
            ]
            batch = [example for (example, _) in explore]

        errors = []
        for example in batch:

            # random UNKing
            for (i, w) in enumerate(example["w"]):
                if w <= 2:
                    continue

                freq = self.fm.word_freq_list[w]
                drop_prob = self.unk_param / (self.unk_param + freq)
                r = np.random.random()
                if r < drop_prob:
                    example["w"][i] = 0

            lstm_outputs = self.evaluate_recurrent(example["w"], example["t"])

            for (left, right), correct in example["struct_data"].items():
                scores = self.evaluate_struct(lstm_outputs, left, right)
                correct = torch.LongTensor([correct]).type_as(example["w"])
                struct_loss = self.criterion(scores, correct)
                errors.append(struct_loss)

            for (left, right), correct in example["label_data"].items():
                scores = self.evaluate_label(lstm_outputs, left, right)
                correct = torch.LongTensor([correct]).type_as(example["w"])
                label_loss = self.criterion(scores, correct)
                errors.append(label_loss)

            batch_error = torch.stack(errors, dim=0).mean()

        return batch_error
