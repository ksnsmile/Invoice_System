# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:40:15 2025

@author: user
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision.transforms import ToTensor
from Preprocessing import resize_bounding_box,normalize_box,resize_and_align_bounding_box

class krri_invoice(Dataset):
    """LayoutLM dataset with visual features."""

    def __init__(self, image_file_names, tokenizer, max_length, target_size, train=True):
        # 이미지 파일만을 선택하기 위해 확장자 필터링 추가
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # 허용할 이미지 확장자
        self.image_file_names = [
            name for name in image_file_names 
            if name.lower().endswith(valid_extensions) and ".ipynb_checkpoints" not in name
        ]
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.target_size = target_size
        self.pad_token_box = [0, 0, 0, 0]
        self.train = train

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):

        # first, take an image
        item = self.image_file_names[idx]
        if self.train:
            base_path = "content_combine/data_combine"
        else:
            base_path = "content_combine/data_test2"

        ########체크포인트 처리 필요x
        # target = base_path + '/.ipynb_checkpoints'
        # if base_path + '/' + item == target:
        #   print('체크포인트 걸림!')
        #   return

        #original_image = Image.open(base_path + '/' + item).convert("L")
        original_image = Image.open(base_path + '/' + item).convert("RGB")

        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))
        
        # first, read in annotations at word-level (words, bounding boxes, labels)
        # with open(base_path + '/annotations/' + item[:-4] + '.json') as f:
        
        with open('content_combine/annotation_combine/' + item.split('.')[0] + '.json', encoding='utf-8') as f:
         origin = json.load(f)
        #data = origin[item.split('.')[0] ]
         data = origin

        words = []
        unnormalized_word_boxes = []
        word_labels = []

        #nrm_to_abs_width = original_image.size[0] / 100
        #nrm_to_abs_height = original_image.size[1] / 100

        for i in data['objects']:
            words.append(i['transcription'])
            word_labels.append(i['label'])
            unnormalized_word_boxes.append([int(i['x']), int(i['y']), int((i['x'] + i['width'])), int((i['y'] + i['height']))])
            
       # for annotation in data['form']:
         # get label
        #  label = annotation['label']
          # get words
       #   for annotated_word in annotation['words']:
        #      if annotated_word['text'] == '':
        #        continue
         #     words.append(annotated_word['text'])
        #      unnormalized_word_boxes.append(annotated_word['box'])
         #     word_labels.append(label)

        width, height = original_image.size
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        assert len(words) == len(normalized_word_boxes)
        #print(len(words))
        #print(len(normalized_word_boxes))
        #print(len(word_labels))

        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        token_boxes = []
        unnormalized_token_boxes = []
        token_labels = []
        for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes, word_labels):
            word_tokens = self.tokenizer.tokenize(word)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            for i in range(len(word_tokens)):
              if i == 0:
                token_labels.extend(['B-' + label])
              else:
                token_labels.extend(['I-' + label])

        # Truncation of token_boxes + token_labels
        special_tokens_count = 2
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
            token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]

        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
        token_labels = [-100] + token_labels + [-100]

        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
        token_labels += [-100] * padding_length
        encoding['bbox'] = token_boxes
        encoding['labels'] = token_labels

        assert len(encoding['input_ids']) == self.max_seq_length
        assert len(encoding['attention_mask']) == self.max_seq_length
        assert len(encoding['token_type_ids']) == self.max_seq_length
        assert len(encoding['bbox']) == self.max_seq_length
        assert len(encoding['labels']) == self.max_seq_length

        encoding['resized_image'] = ToTensor()(resized_image)
        # rescale and align the bounding boxes to match the resized image size (typically 224x224)
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size)
                                                          for bbox in unnormalized_token_boxes]

        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes

        # finally, convert everything to PyTorch tensors
        for k,v in encoding.items():
            if k == 'labels':
              label_indices = []
              # convert labels from string to indices
              for label in encoding[k]:
                if label != -100:
                  label_indices.append(label2idx[label])
                else:
                  label_indices.append(label)
              encoding[k] = label_indices
            encoding[k] = torch.as_tensor(encoding[k])

        return encoding
    
labels = ['B-sign', 'I-sign',
'B-recipient_key', 'I-recipient_key',
'B-recipient_name', 'I-recipient_name',
'B-recipient_phone_number_key', 'I-recipient_phone_number_key',
'B-recipient_phone_number', 'I-recipient_phone_number',
'B-recipient_address_do', 'I-recipient_address_do',
'B-recipient_address_si', 'I-recipient_address_si',
'B-recipient_address_gun', 'I-recipient_address_gun',
'B-recipient_address_gu', 'I-recipient_address_gu',
'B-recipient_address_eup', 'I-recipient_address_eup',
'B-recipient_address_myeon', 'I-recipient_address_myeon',
'B-recipient_address_ri', 'I-recipient_address_ri',
'B-recipient_address_dong', 'I-recipient_address_dong',
'B-recipient_address_jibeon', 'I-recipient_address_jibeon',
'B-recipient_address_ro_name', 'I-recipient_address_ro_name',
'B-recipient_address_gil_name', 'I-recipient_address_gil_name',
'B-recipient_address_ro_number', 'I-recipient_address_ro_number',
'B-recipient_address_building_number', 'I-recipient_address_building_number',
'B-recipient_address_room_number', 'I-recipient_address_room_number',
'B-recipient_address_detail', 'I-recipient_address_detail',
'B-sender_key', 'I-sender_key',
'B-sender_name', 'I-sender_name',
'B-sender_phone_number_key', 'I-sender_phone_number_key',
'B-sender_phone_number', 'I-sender_phone_number',
'B-sender_address_do', 'I-sender_address_do',
'B-sender_address_si', 'I-sender_address_si',
'B-sender_address_gun', 'I-sender_address_gun',
'B-sender_address_gu', 'I-sender_address_gu',
'B-sender_address_eup', 'I-sender_address_eup',
'B-sender_address_myeon', 'I-sender_address_myeon',
'B-sender_address_ri', 'I-sender_address_ri',
'B-sender_address_dong', 'I-sender_address_dong',
'B-sender_address_jibeon', 'I-sender_address_jibeon',
'B-sender_address_ro_name', 'I-sender_address_ro_name',
'B-sender_address_gil_name', 'I-sender_address_gil_name',
'B-sender_address_ro_number', 'I-sender_address_ro_number',
'B-sender_address_building_number', 'I-sender_address_building_number',
'B-sender_address_room_number', 'I-sender_address_room_number',
'B-sender_address_detail', 'I-sender_address_detail',
'B-volume_key', 'I-volume_key',
'B-volume', 'I-volume',
'B-delivery_message_key', 'I-delivery_message_key',
'B-delivery_message', 'I-delivery_message',
'B-product_name_key', 'I-product_name_key',
'B-product_name', 'I-product_name',
'B-tracking_number_key', 'I-tracking_number_key',
'B-tracking_number', 'I-tracking_number',
'B-weight_key', 'I-weight_key',
'B-weight', 'I-weight',
'B-terminal_number', 'I-terminal_number',
'B-company_name', 'I-company_name',
'B-handwriting', 'I-handwriting',
'B-others', 'I-others'
]


idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}