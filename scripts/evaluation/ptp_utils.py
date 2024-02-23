# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict
from einops import rearrange, repeat

def register_attention_control(model, controller):
    def block_forward(block, place_in_unet):
        original_forward = block.forward

        def forward(*args, **kwargs):
            context = args[0] if args else kwargs.get('context', None)
            encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
            attention_mask = kwargs.get('attention_mask', None)
            video_length = kwargs.get('video_length', None)
            
            # Preprocess with controller before calling original forward
            norm_hidden_states = block.norm1(context)
            norm_hidden_states, k_input, v_input = controller(
                norm_hidden_states, video_length, place_in_unet
            )
            
            # Modify kwargs for attention inputs
            kwargs['k_input'] = k_input
            kwargs['v_input'] = v_input
            kwargs['attention_mask'] = attention_mask
            kwargs['video_length'] = video_length
            
            # Call the original forward method with modified arguments
            return original_forward(*args, **kwargs)

        return forward

    def register_recur(net, count, place_in_unet):
        for child_name, child_module in net.named_children():
            if getattr(child_module, 'initialized_from', '') == "SpatialTransformer":
                child_module.forward = block_forward(child_module, place_in_unet)
                count += 1
            else:
                count = register_recur(child_module, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net_module in model.named_children():
        if "model" in net_name:
            cross_att_count += register_recur(net_module, 0, "model")
        # Add other conditions if there are other specific parts of the model you wish to modify
    controller.num_att_layers = cross_att_count * 2


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[
        float, Tuple[float, float], Dict[str, Tuple[float, float]]
    ],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )  # time, batch, heads, pixels, words
    return alpha_time_words