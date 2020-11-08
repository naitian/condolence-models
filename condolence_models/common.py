import logging
import os
import shutil
import tempfile

from itertools import islice

import requests
import torch
from tqdm import tqdm
from .bert_classifier.classifier import BertClassifier
from .bert_classifier.utils import preprocess


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def load_model(model_type, path, device="cpu"):
    if not os.path.isfile(path):
        logger.info(f'Model {model_type} does not exist at {path}. Attempting to download it.')
        model = f'{model_type}_model'
        fetch_pretrained_model(model, path)
    model = BertClassifier(2)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def fetch_pretrained_model(model, model_path):
    PRETRAINED_MODEL_ARCHIVE_MAP = {
        'condolence_model': ['http://jurgens.people.si.umich.edu/models/condolence_model.pth'],
        'distress_model': ['http://jurgens.people.si.umich.edu/models/distress_model.pth'],
    }
    assert model in PRETRAINED_MODEL_ARCHIVE_MAP
    model_urls = PRETRAINED_MODEL_ARCHIVE_MAP[model]
    model_urls = PRETRAINED_MODEL_ARCHIVE_MAP[model]

    download_flag = False
    for idx, model_url in enumerate(model_urls):
        try:
            temp_file = tempfile.NamedTemporaryFile()
            logger.info(f'{model_path} not found in cache, downloading from {model_url} to {temp_file.name}')

            req = requests.get(model_url, stream=True)
            content_length = req.headers.get('Content-Length')
            total = int(content_length) if content_length is not None else None
            progress = tqdm(unit="KB", total=round(total / 1024))
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(1)
                    temp_file.write(chunk)
            progress.close()
            temp_file.flush()
            temp_file.seek(0)
            download_flag = True
            break
        except Exception as e:
            logger.warning(f'Download from {idx + 1}/{len(model_urls)} mirror failed with an exception of\n{str(e)}')
            try:
                temp_file.close()
            except Exception as e_file:
                logger.warning(f'temp_file failed with an exception of \n{str(e_file)}')
            continue

    if not download_flag:
        logging.warning(f'Download from all mirrors failed. Please retry.')
        return

    logger.info(f'Model {model} was downloaded to a tmp file.')
    logger.info(f'Copying tmp file to {model_path}.')
    with open(model_path, 'wb') as cache_file:
        shutil.copyfileobj(temp_file, cache_file)
    logger.info(f'Copied tmp model file to {model_path}.')
    temp_file.close()


def batch_generator(iterable, tokenizer, batch_size=16, device="cpu"):
    ct = iterable
    piece = islice(ct, batch_size)
    while piece:
        input_list = []
        for text in piece:
            tokens = tokenizer.tokenize(preprocess(text))
            tokens = tokens[:128]
            indexed_toks = tokenizer.convert_tokens_to_ids(tokens)
            indexed_toks += [0] * (128 - len(indexed_toks))
            input_list.append(torch.tensor(indexed_toks).unsqueeze(0))
        if len(input_list) == 0:
            return
        batch_inputs = torch.cat(input_list, 0)
        yield batch_inputs.to(device)
        piee = islice(ct, batch_size)
