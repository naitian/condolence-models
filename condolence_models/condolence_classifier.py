import logging
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from .common import load_model, fetch_pretrained_model, batch_generator, logger
from pytorch_transformers import BertTokenizer


class CondolenceClassifier():
    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batch_size=16):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_type = "condolence"
        self.batch_size = batch_size

        model_path = os.path.join(os.path.dirname(__file__), f"models/{self.model_type}.pth")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = load_model(self.model_type, model_path, self.device)

    def predict(self, text):
        if isinstance(text, str):
            text = [text]
        scores = torch.tensor([], requires_grad=False)
        for i, inputs in enumerate(
                tqdm(batch_generator(
                    text,
                    tokenizer=self.tokenizer,
                    batch_size=self.batch_size,
                    device=self.device
                ))
        ):
            scores = torch.cat((
                scores,
                nn.functional.softmax(self.model(inputs), dim=1)[:, 1].cpu()
            ), 0)
        return scores.detach().numpy()
