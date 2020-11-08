import os
import shutil

import numpy as np
import pandas as pd
import torch

from .common import fetch_pretrained_model, logger

from simpletransformers.classification import ClassificationModel

manual_seed = 1234
train_args={
    "regression": True,
    "manual_seed": manual_seed,
}

def fetch_simple_transformer_model(model, model_folder, model_path):
    PRETRAINED_MODEL_ARCHIVE_MAP = {
        'empathy_model': ['http://jurgens.people.si.umich.edu/models/empathy_model.tar.gz'],
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

def unzip_simple_transformer_model(model, model_folder, model_path):
    logger.info('Extracting file to folder')
    shutil.unpack_archive(model_path, model_folder)

class EmpathyClassifier():
    def __init__(self,
                 use_cuda=torch.cuda.is_available(),
                 cuda_device=0,
                 batch_size=16):
        self.model_type = "distress"
        train_args["eval_batch_size"] = batch_size

        model_path = os.path.join(os.path.dirname(__file__), "models/empathy/")
        model_file = os.path.join(os.path.dirname(__file__), "models/empathy.tar.gz")
        if not os.path.isdir(model_path):
            model = f'{self.model_type}_model'
            if not os.path.isfile(model_file):
                logger.info(f'Model {self.model_type} does not exist at {model_path}. Attempting to download it.')
                fetch_pretrained_model(model, model_path, model_file)
            unzip_simple_transformer_model(model, model_path, model_file)

        # Create a ClassificationModel
        self.model = ClassificationModel(
            'roberta',
            model_path,
            num_labels=1,
            use_cuda=use_cuda,
            cuda_device=cuda_device,
            args=train_args
        )

    def predict(self, text):
        if isinstance(text[0], str):
            text = [text]
        predictions, raw_outputs = self.model.predict(text)
        return raw_outputs
