# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import copy
import os

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor

top_k=10

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



def len_list(x,n):
    return [i for i in x if len(i)<=n]

class Model(nn.Module):
    def __init__(self,config,load_emb=True,path=None,num_res_layer=0,lm_head=False):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.use_emb = load_emb

        if load_emb:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            from safetensors import safe_open
            try:
                with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
                    index_json=json.loads(f.read())
                    emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path,emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights=torch.load(os.path.join(path,emb_path))
                tensor=weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor
            from .cross_net import CrossNetV2
            self.fc = CrossNetV2(2 * config.hidden_size, config.hidden_size, 1)
            for param in self.embed_tokens.parameters():
                param.requires_grad = False
        # else:
        #     self.fc = nn.Linear(config.hidden_size, config.hidden_size)


        from mamba_ssm import Mamba
        from .mamba import MambaConfig
        self.m_config = MambaConfig(d_model=config.hidden_size,n_layers=1)
        self.layers = nn.ModuleList([Mamba(d_model=config.hidden_size) for _ in range(self.m_config.n_layers)])
        # self.layers = DIN(seq{0, t-1}, t) # ETA, SDIM [B, Seq_len, emb] [B, Seq_len, emb]
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # self.fc=nn.Linear(2*config.hidden_size,config.hidden_size,bias=bias)
        if num_res_layer>0:
            self.fc_out = nn.Sequential(
                *([ResBlock(config.hidden_size)] * num_res_layer)
            )
        if lm_head:
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        self.head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        try:
            with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(os.path.join(path, head_path),
                           framework="pt",
                           device="cpu") as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(path, head_path))
            tensor = weights["lm_head.weight"].float()

        self.head.weight.data = tensor
        for param in self.head.parameters():
            param.requires_grad = False

        self.head.eval()

    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffers(self.tree,self.embed_tokens.weight.device)


    def reset(self):
        self.tree_mask=None


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min


        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if self.use_emb:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)
                #inputs_embeds = inputs_embeds.detach()

            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # else:
        #     hidden_states = self.fc(hidden_states)

        next_decoder_cache = () if use_cache else None

        # if not past_key_values:
        #
        #     caches = [(None, torch.zeros(1, self.m_config.d_inner, self.m_config.d_conv - 1, device=input_ids.device))
        #               for _ in range(self.m_config.n_layers)]
        # else:
        #     caches = past_key_values

        for idx, decoder_layer in enumerate(self.layers):
            # 训练用forward，inference用step

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states
                    # attention_mask,
                    # position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states
                    # attention_mask=attention_mask,
                    # position_ids=position_ids,
                    # past_key_value=past_key_value,
                    # output_attentions=output_attentions,
                    # use_cache=use_cache,
                )

            hidden_states = layer_outputs
            # hidden_states = layer_outputs[0]
            if getattr(self,"fc_out"):
                hidden_states = self.fc_out(hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        if use_cache:
            return hidden_states,next_decoder_cache

        return hidden_states

    @torch.no_grad()
    def step(self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None):
        try:
            batch_size, seq_length = hidden_states.shape
        except:
            batch_size, seq_length,_ = hidden_states.shape
        # seq_length_with_past = seq_length
        # past_key_values_length = 0

        if self.use_emb:

            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)
                # inputs_embeds = inputs_embeds.detach()

            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # else:
        #     hidden_states = self.fc(inputs_embeds)

        if len(hidden_states.shape)==3 and hidden_states.shape[0]==1:
            hidden_states = hidden_states.squeeze(0)

        next_decoder_cache = [[],[]] if use_cache else None

        if not past_key_values:

            # conv_state = [torch.zeros(
            #     batch_size,
            #     self.m_config.d_inner,
            #     self.m_config.d_conv,
            #     device=self.conv1d.weight.device,
            #     dtype=self.conv1d.weight.dtype,
            # ) for _ in range(self.m_config.n_layers)]
            # ssm_state = [torch.zeros(
            #     batch_size,
            #     self.m_config.d_inner,
            #     self.m_config.d_state,
            #     device=self.dt_proj.weight.device,
            #     dtype=self.dt_proj.weight.dtype,
            #     # dtype=torch.float32,
            # ) for _ in range(self.m_config.n_layers)]
            #
            # caches = [conv_state, ssm_state]
            caches = [
                [
                    torch.zeros(
                        batch_size,
                        self.m_config.d_inner,
                        self.m_config.d_conv,
                        device=self.head.weight.device,
                        dtype=self.head.weight.dtype,
                    ),
                    torch.zeros(
                        batch_size,
                        self.m_config.d_inner,
                        self.m_config.d_state,
                        device=self.head.weight.device,
                        dtype=self.head.weight.dtype,
                        # dtype=torch.float32,
                    )
                ] for _ in range(self.m_config.n_layers)
            ]
        else:
            caches = past_key_values

        for idx, decoder_layer in enumerate(self.layers):

            # 训练用forward，inference用step

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module.step(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,caches[idx][0], caches[idx][1]
                    # attention_mask,
                    # position_ids,
                )
            else:
                layer_outputs = decoder_layer.step(hidden_states.unsqueeze(1),caches[idx][0], caches[idx][1])

            hidden_states = layer_outputs[0]
            if getattr(self,"fc_out"):
                hidden_states = self.fc_out(hidden_states)

            if use_cache:
                next_decoder_cache[0].append(layer_outputs[1])
                next_decoder_cache[1].append(layer_outputs[2])

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    @torch.no_grad()
    def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
        return_input_ids=copy.deepcopy(input_ids[0].tolist())
        input_ids=input_ids[:,1:]

        #input_ids=input_ids.to(hidden_states.device)
        if use_cache:
            past_key_values=None
            for i in range(max_length):
                if past_key_values!=None:
                    out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
                else:
                    out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                return_input_ids.append(token.item())
                if token == 2:
                    break
                #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
        else:
            for i in range(max_length):
                out_hidden=self(hidden_states,input_ids=input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                return_input_ids.append(token.item())
                input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                if token==2:
                    break
                hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

        return return_input_ids

    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.stable_kv=None

    @torch.no_grad()
    def repeat_hidden(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
        return torch.cat(new_hidden,dim=1)

    def repeat_caches(self, caches:List[Tuple[torch.Tensor,torch.Tensor]], repeat_num):
        new_caches = []
        for j in range(self.m_config.n_layers):
            new_h_layer = []
            new_input_layer = []
            for id, i in enumerate(repeat_num):
                new_h_layer.append(caches[j][0][id].repeat(i,1,1))
                new_input_layer.append(caches[j][1][id].repeat(i, 1, 1))
            new_caches.append([torch.cat(new_h_layer,dim=0), torch.cat(new_input_layer,dim=0)])
        return new_caches

    # @torch.no_grad()
    # def sample(self,tensor,k=1,replacement=True):
    #     probabilities = torch.nn.functional.softmax(tensor, dim=1)
    #     sampled_indices = torch.multinomial(probabilities, k,replacement=replacement)
    #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
    #
    #     return  sampled_indices,sampled_probs

    def sample(self,logits, logits_processor,k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs,probabilities

    @torch.no_grad()
    def topK_generate(self, hidden_states, input_ids, head, logits_processor,max_length=4, use_cache=True):
        # test_=input_ids
        # input_ids = torch.tensor([state[1:]])
        batch_size, seqlen, dim = hidden_states.shape
        # input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token,ss_prob,ss_op = [],[],[]
        len_posi=input_ids.shape[1]
        self.reset()
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            # kv_len=self.stable_kv[0][0].shape[2]
            caches = self.stable_kv
            for i in range(seqlen):
                out_hidden, caches = self.step(hidden_states[:,i], input_ids=input_ids[:,-(seqlen-i)], past_key_values=caches,use_cache=True)
        else:
            # conv_state = [torch.zeros(
            #     batch_size,
            #     self.m_config.d_inner,
            #     self.m_config.d_conv,
            #     device=self.head.weight.device,
            #     dtype=self.head.weight.dtype,
            # ) for _ in range(self.m_config.n_layers)]
            # ssm_state = [torch.zeros(
            #     batch_size,
            #     self.m_config.d_inner,
            #     self.m_config.d_state,
            #     device=self.head.weight.device,
            #     dtype=self.head.weight.dtype,
            #     # dtype=torch.float32,
            # ) for _ in range(self.m_config.n_layers)]

            caches = [
                [
                    torch.zeros(
                        batch_size,
                        self.m_config.d_inner,
                        self.m_config.d_conv,
                        device=self.head.weight.device,
                        dtype=self.head.weight.dtype,
                    ),
                    torch.zeros(
                        batch_size,
                        self.m_config.d_inner,
                        self.m_config.d_state,
                        device=self.head.weight.device,
                        dtype=self.head.weight.dtype,
                        # dtype=torch.float32,
                    )
                ] for _ in range(self.m_config.n_layers)
            ]

            cold_start_length = 0# hidden_states.shape[1]
            for i in range(cold_start_length, hidden_states.shape[1]):
                out_hidden, caches = self.step(hidden_states[:,i], input_ids=input_ids[:,i], past_key_values=caches,use_cache=True)
        self.stable_kv=caches

        # last_hidden = out_hidden[:, -1]
        last_hidden = out_hidden.squeeze(1)
        last_headout = head(last_hidden)


        for i in range(len(self.tree_buffer['tree_indices'])):
            if logits_processor is not None:
                topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
            else:
                top=torch.topk(last_headout, top_k, dim=-1)
                topk_index,topk_prob = top.indices,top.values
                op=None

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            #topk_index = torch.topk(last_headout, top_k, dim=-1).indices
            topk_index = topk_index.view(-1)
            select_index=topk_index[self.tree_buffer['tree_indices'][i]]
            #len_sq=select_index.shape[0]
            input_ids=select_index[None,:]
            # if i==0:
            #     hidden_states = out_hidden[:, -1:]
            # else:
            hidden_states=self.repeat_hidden(out_hidden,self.tree_buffer["repeat_nums"][i])
            # hidden_states = hidden_states.squeeze(0)
            caches_repeat = self.repeat_caches(caches, self.tree_buffer["repeat_nums"][i])
            # caches_repeat = [(caches[t][0].repeat(hidden_states.shape[1],1,1), caches[t][1].repeat(hidden_states.shape[1],1,1)) for t in range(len(caches))]
            #hidden_states = hidden_states.repeat(1,len_sq,1)
            self.tree_mask=self.tree_buffer['attn_mask'][i]
            # position_ids=len_posi+self.tree_buffer["position_ids"][i]
            # out_hiddens, cachess = [], []

            out_hidden, caches = self.step(hidden_states, input_ids=input_ids, past_key_values=caches_repeat, use_cache=True)

            # TO DO 用并发替换掉for，避免cat和append操作
            # for k in range(input_ids.shape[1]):
            #     out_hidden, caches = self.step(hidden_states[:,k], input_ids=input_ids[:,k], past_key_values=[caches_repeat[k]],position_ids=position_ids,use_cache=True)
            #     out_hiddens.append(out_hidden)
            #     cachess += caches
            len_posi += 1
            # out_hidden = torch.cat(out_hiddens,dim=0)
            # caches = cachess
            out_hidden = out_hidden.transpose(0,1)
            last_hidden = out_hidden.squeeze(0)
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(last_hidden, self.headweight)
            #last_headout = head(out_hidden[0])
            #sslogits.append(last_headout)
            #print(select_index)

        if logits_processor is not None:
            topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
        else:
            top = torch.topk(last_headout, top_k, dim=-1)
            topk_index, topk_prob = top.indices, top.values
            op=None
        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)


        return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)




    @torch.no_grad()
    def acc(self,data,head,max_length=5):
        hidden_states=data["hidden_states"]
        input_ids=data["input_ids"]
        #attention_mask=data["attention_mask"]
        loss_mask=data["loss_mask"]
        sample_mask=data["sample_mask"]
        target=data["target"]
        total=[0 for _ in range(max_length)]
        correct=[0 for _ in range(max_length)]
        bs,sl=hidden_states.shape[0],hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout=head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i,j]==0:
                    continue
                single_hidden_states=hidden_states[i,:j]
                single_input_ids=input_ids[i,:j]


                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i,single_hidden_states.shape[1]-1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token=input_ids[i,single_hidden_states.shape[1]-1]
                    tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                    if not (target_in_token==tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token==target_out_token:
                        correct[k]+=1
                    else:
                        for kk in range(k,max_length):
                            total[kk]+=1
                        break

                    single_hidden_states=torch.cat((single_hidden_states,out_hidden[:,-1:]),dim=1)
                    single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)


        acc=[correct[i]/total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):
    def __init__(self,ins=6566,outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins,outs,bias=False)
    def forward(self,x):
        return self.fc(x)



import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__=="__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config,load_emb=True,path="/home/lyh/weights/hf/vicuna_v13/7B/")
    print(model)
