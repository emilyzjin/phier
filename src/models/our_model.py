import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image 

from .base_model import EndToEndModel

class ObjectCentricModel(EndToEndModel):
    def __init__(self, args=None): 
        super(ObjectCentricModel, self).__init__(args) 
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.vision_encoder = self.clip_model.vision_model
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0)
        self.conv1.weight.data = self.clip_model.vision_model.encoder.layers[-1].self_attn.v_proj.weight.data.unsqueeze(-1).unsqueeze(-1) # value encoder in the QKV-attention layer
        self.conv2.weight.data = self.clip_model.visual_projection.weight.data.unsqueeze(-1).unsqueeze(-1) # last linear layer
 

    def get_obj_text_embeddings(self, objs):
        '''
        Given a list of objects, get the text embeddings for each object using the CLIP text encoder
        objs: list of strings, each string is the name of an object
        Return: a tensor where each row is the text embedding for an object
        '''
        inputs = [f'A photo of a {obj} in the scene' if obj != '' else '' for obj in objs]
        # inputs = objs
        inputs = self.clip_tokenizer(inputs, padding=True, return_tensors="pt").to(self.device)
        text_embeddings = self.clip_model.get_text_features(**inputs)
        text_embeddings = text_embeddings.reshape(len(objs) // 2, 2, text_embeddings.shape[-1])
        return text_embeddings

    def apply_masks(self, images, obj_masks, max_len=64):
        global_masks = obj_masks.max(dim=1, keepdim=True)[0] # (batch_size, num_obj, 7, 7)
        masked_images = images * global_masks # (batch_size, 768)
        return masked_images

    def get_clip_image_embeddings(self, images):
        # use clip image encoder to get image embeddings
        image_embeddings = self.vision_encoder(images)  # Get image embedding for the object
        image_embeddings = image_embeddings[0][:, :-1, :]  # all patch embeddings except the CLS token. shape (batch_size, num_patches, 768)
        image_embeddings = image_embeddings.reshape(image_embeddings.shape[0], 7, 7, 768) # (batch_size, 7, 7, 768)
        return image_embeddings

    def get_obj_masks(self, image_embeds, text_embeds):  
        # apply 1x1 conv layers to image features
        batch_size = image_embeds.shape[0]
        image_embeds = image_embeds.permute(0, 3, 1, 2) # (batch_size, num_channels, h, w)
        image_embeds = self.conv2(image_embeds) # (batch_size, 768, 7, 7)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # normalize 

        text_embeds = text_embeds.view(-1, 512) 
        text_embeds = text_embeds.unsqueeze(-1).unsqueeze(-1) # (batch_size * 2, 512, 1, 1)

        image_embeds = image_embeds.unsqueeze(1).repeat(1, 2, 1, 1, 1) # (batch_size, 2, 512, 7, 7)
        image_embeds = image_embeds.view(batch_size * 2, 512, 7, 7)
        image_embeds = image_embeds.view(1, batch_size * 2 * 512, 7, 7) 

        output = F.conv2d(image_embeds, text_embeds, groups=batch_size*2) # 1 x batch_size * 2 x 7 x 7
        output = output.view(batch_size*2, 1, 7, 7) # (batch_size * 2, 7, 7)

        mask = F.interpolate(output, size=(224, 224), mode='bicubic', align_corners=False) # (batch_size, 1, 224, 224)
        mask_min = mask.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        mask_max = mask.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        mask = (mask - mask_min) / (mask_max - mask_min) # normalize to [0, 1] per image in batch
        mask = mask.view(batch_size, 2, 224, 224)

        return mask, output 

    def get_obj_conditioned_image_embedding(self, images, objss):
        text_embeds = self.get_obj_text_embeddings(objss) # shape (B x 2 x 512)
        image_embeds = self.get_clip_image_embeddings(images) # shape (B x 768)
        obj_masks, mask_embeds = self.get_obj_masks(image_embeds, text_embeds) # shape (B x 2 x 512)
        
        # apply masks to images
        masked_images = self.apply_masks(images, obj_masks)  # Apply object masks to each image
        return masked_images, obj_masks, mask_embeds

    def get_conditioned_embedding(self, images, states, save_masks=False):
        timestamp = int(time.time())

        preds, objss = self.parse_query_states(states)

        # concatenate the full query embedding w/ the aligned object embeddings
        masked_images, obj_masks, mask_embeds = self.get_obj_conditioned_image_embedding(images, objss)

        if save_masks:
            os.makedirs(os.path.join('masked_imgs', self.dataset), exist_ok=True)
            for i in range(len(states)):
                state = states[i]
                obj_mask = masked_images[i:i+1]
                image = images[i]
                masked_image = masked_images[i]
                image_pil = to_pil_image(image)
                masked_image_pil = to_pil_image(masked_image) 
                
        # Concatenate scene and state embeddings
        image_embeds = self.get_image_embedding(masked_images)
        pred_embeds = self.get_query_embedding(preds, mode='predicate')
        conditioned_embedding = torch.cat((pred_embeds, image_embeds), dim=1)

        return conditioned_embedding 

