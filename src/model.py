import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.build()

    def build(self):
        self.period_embed = nn.Linear(48, 32)
        self.weekday_embed = nn.Linear(7, 4)
        self.day_embed = nn.Linear(31, 16)
        self.month_embed = nn.Linear(12, 8)
        self.peak_embed = nn.Linear(15, 8)
        self.special_day_embed = nn.Linear(13, 8)
        self.street_type_embed = nn.Linear(53, 32)
        self.street_level_embed = nn.Linear(4, 2)

        self.dense_layer_1 = nn.Linear(110, 128)
        self.activation_1 = nn.Tanh()
        self.dense_layer_2 = nn.Linear(128, 64)
        self.activation_2 = nn.Tanh()
        self.dense_layer_3 = nn.Linear(64, 32)
        self.activation_3 = nn.Tanh()

        self.dense_layer_4 = nn.Linear(32+5, 32)
        self.activation_4 = nn.LeakyReLU()
        self.dense_layer_5 = nn.Linear(32, 16)
        self.activation_5 = nn.LeakyReLU()
        self.dense_layer_6 = nn.Linear(16, 8)
        self.activation_6 = nn.LeakyReLU()
        self.dense_layer_final = nn.Linear(8, 6)

    def forward(self, X):
        special_day, peak, month, day, weekday, period, street_level, street_type, rest = \
            X[:, 0:13], X[:, 13:28], X[:, 28:40], X[:, 40:71], X[:,
                                                                 71:78], X[:, 78:126], X[:, 126:130], X[:, 130:183], X[:, 183:]

        period_embed = self.period_embed(period)
        weekday_embed = self.weekday_embed(weekday)
        day_embed = self.day_embed(day)
        month_embed = self.month_embed(month)
        peak_embed = self.peak_embed(peak)
        special_day_embed = self.special_day_embed(special_day)
        street_type_embed = self.street_type_embed(street_type)
        street_level_embed = self.street_level_embed(street_level)

        out_1 = self.dense_layer_1(torch.cat((period_embed, weekday_embed, day_embed, month_embed,
                                   peak_embed, special_day_embed, street_type_embed, street_level_embed), axis=1))
        out_1 = self.activation_1(out_1)
        out_2 = self.dense_layer_2(out_1)
        out_2 = self.activation_2(out_2)
        out_3 = self.dense_layer_3(out_2)
        out_3 = self.activation_3(out_3)

        out_4 = self.dense_layer_4(torch.cat((rest, out_3), axis=1))
        out_4 = self.activation_4(out_4)
        out_5 = self.dense_layer_5(out_4)
        out_5 = self.activation_5(out_5)
        out_6 = self.dense_layer_6(out_5)
        out_6 = self.activation_6(out_6)
        out_7 = self.dense_layer_final(out_6)

        return out_7
