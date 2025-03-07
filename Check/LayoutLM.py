# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:41:04 2025

@author: user
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIAlign
from transformers import LayoutLMModel
from transformers.modeling_outputs import TokenClassifierOutput


class LayoutLMForTokenClassification(nn.Module):
    def __init__(self, output_size=(3,3), #1,9216
                 spatial_scale=14/1024,
                 sampling_ratio=2,
                 dropout_prob=0.1
        ):
        super().__init__()

        # LayoutLM base model + token classifier
        self.num_labels = len(label2idx)
        self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=self.num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        #self.dropout = nn.Dropout(self.layoutlm.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, self.num_labels)

        # backbone + roi-align + projection layer
        model = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-3]))
        #self.backbone = nn.Sequential(*(list(model.children())[:0]))

        self.roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        self.projection = nn.Linear(in_features=1024*3*3, out_features=self.layoutlm.config.hidden_size)

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask,
        token_type_ids,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        resized_images=None, # shape (N, C, H, W), with H = W = 224
        resized_and_aligned_bounding_boxes=None, # single torch tensor that also contains the batch index for every bbox at image size 224
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:torch.LongTensor of shape :obj:(batch_size, sequence_length), optional):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels -
            1]`.

        """
        return_dict = return_dict if return_dict is not None else self.layoutlm.config.use_return_dict

        # first, forward pass on LayoutLM
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # next, send resized images of shape (batch_size, 3, 224, 224) through backbone to get feature maps of images
        # shape (batch_size, 1024, 14, 14)
        feature_maps = self.backbone(resized_images)

        # next, use roi align to get feature maps of individual (resized and aligned) bounding boxes
        # shape (batch_size*seq_len, 1024, 3, 3)
        device = input_ids.device
        resized_bounding_boxes_list = []
        for i in resized_and_aligned_bounding_boxes:
          resized_bounding_boxes_list.append(i.float().to(device))

        feat_maps_bboxes = self.roi_align(input=feature_maps,
                                        # we pass in a list of tensors
                                        # We have also added -0.5 for the first two coordinates and +0.5 for the last two coordinates,
                                        # see https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
                                        rois=resized_bounding_boxes_list
                           )

        # next, reshape  + project to same dimension as LayoutLM.
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        feat_maps_bboxes = feat_maps_bboxes.view(batch_size, seq_len, -1) # Shape (batch_size, seq_len, 1024*3*3)
        projected_feat_maps_bboxes = self.projection(feat_maps_bboxes) # Shape (batch_size, seq_len, hidden_size)

        # add those to the sequence_output - shape (batch_size, seq_len, hidden_size)
        sequence_output += projected_feat_maps_bboxes

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
    
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