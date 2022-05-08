import torch
from transformers import BertTokenizer
from PIL import Image
import argparse
import os

from models import caption
from datasets import coco, utils

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def predict(image_path):
  model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
  end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
  start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
  end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
  caption, cap_mask = create_caption_and_mask(start_token, 128)

  image = Image.open(image_path)
  image = coco.val_transform(image)
  image = image.unsqueeze(0)


  model.eval()
  for i in range(128 - 1):
      predictions = model(image, caption, cap_mask)
      predictions = predictions[:, i, :]
      predicted_id = torch.argmax(predictions, axis=-1)

      if predicted_id[0] == 102:
          return tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)

      caption[:, i+1] = predicted_id[0]
      cap_mask[:, i+1] = False
  return tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
