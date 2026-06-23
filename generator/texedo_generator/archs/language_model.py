import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb
import pdb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,                   
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",        # new_token
        motion_codebook_size: int = 512,       #codebook size
        framerate: float = 20.0,               #fps
        down_t: int = 2,                       #down_t
        predict_ratio: float = 0.2,            #predict_ratio
        inbetween_ratio: float = 0.25,
        max_length: int = 256,                 #max_length
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(
                model_path)
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)])

        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head

        # Lora
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            from peft.utils.other import fsdp_auto_wrap_policy
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                #  inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)

        # Sampling parameters for generation diversity
        self._sample_temperature = 1.0
        self._sample_top_k = 0
        self._sample_top_p = 1.0

    def set_sampling_params(self, temperature=1.0, top_k=0, top_p=1.0):
        """Set sampling parameters used during generate_conditional."""
        self._sample_temperature = temperature
        self._sample_top_k = top_k
        self._sample_top_p = top_p

    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):

        # print("\n" + "="*80)
        # print("FORWARD_ENCDEC - BATCH DEBUG INFO")
        # print("="*80)
        # print(f"Batch size: {len(texts)}")
        # print(f"Motion tokens shape: {motion_tokens.shape}")
        # print(f"Lengths: {lengths[:3]}...") # Show first 3
        
        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        # print(f"\nMotion strings (first sample): {motion_strings[0][:200]}...") # Show first 200 chars

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        # print(f"\nCondition selected: {condition}")
        
        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            # print(f"\n--- TEMPLATE FULFILLED DATA (first sample) ---")
            # print(f"Task: {tasks[0]}")
            # print(f"Input text: {texts[0][:100]}...")
            # print(f"Template INPUT:  {inputs[0][:200]}...")
            # print(f"Template OUTPUT: {outputs[0][:200]}...")

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='longest',
                                         truncation=True,
                                        return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(
            motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)
        
        # print(f"\n--- TOKENIZATION (first sample) ---")
        # print(f"Source input_ids shape: {source_input_ids.shape}")
        # print(f"Source input_ids (first 20): {source_input_ids[0][:20].tolist()}")
        # print(f"Decoded source input: {self.tokenizer.decode(source_input_ids[0][:50], skip_special_tokens=False)}")

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='longest',
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                motion_tokens.device)
            
            # print(f"\n--- TARGET/LABELS (first sample) ---")
            # print(f"Labels input_ids shape: {labels_input_ids.shape}")
            # print(f"Labels input_ids (first 20): {labels_input_ids[0][:20].tolist()}")
            # print(f"Decoded labels: {self.tokenizer.decode(labels_input_ids[0][:50], skip_special_tokens=False)}")

        labels_input_ids[labels_input_ids == 0] = -100
        
        # print(f"\n--- TEACHER FORCING INPUT TO MODEL ---")
        # print(f"Encoder input_ids shape: {source_input_ids.shape}")
        # print(f"Decoder labels shape: {labels_input_ids.shape}")
        print(f"Labels (-100 for padding): {labels_input_ids[0][:20].tolist()}")
        # print(f"Source attention mask sum: {source_attention_mask[0].sum().item()}")
        
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )
    
        # print(f"\n--- MODEL OUTPUT ---")
        print(f"Loss: {outputs.loss.item():.4f}")
        # print(f"Logits shape: {outputs.logits.shape}")
        # print("="*80 + "\n")

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        condition = random.choice(
            ['text', 'motion', 'supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + ' \n ' + outputs[i] +
                              self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='longest',
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=lables_attention_mask,
                                      labels=inputs["input_ids"])

        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None,
                        temperature: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 1.0):

        # Device
        self.device = self.language_model.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]

        source_encoding = self.tokenizer(texts,
                                         padding='longest',
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        # print("source_input_ids312:", source_input_ids)
        # print("source_attention_mask313:", source_attention_mask)

        # -------------------------------------------------------------------
        # GENERATION PRECISION FIX: Disable AMP autocast during .generate()
        #
        # Problem:  When training with mixed precision (fp16 or bf16), PyTorch
        #           Lightning wraps the entire validation_step in an autocast
        #           context. Inside that context, linear layers produce fp16/bf16
        #           outputs. For T5, softmax over a large vocabulary (50k+ tokens)
        #           in fp16 is numerically unstable — some logit differences that
        #           are small in fp32 collapse to zero in fp16, producing NaN/inf
        #           probabilities. torch.multinomial then raises:
        #             "probability tensor contains inf, nan or element < 0"
        #
        # Fix:      Wrap .generate() in `torch.amp.autocast('cuda', enabled=False)`
        #           to force float32 computation ONLY for the generation pass.
        #           Training forward/backward still runs in bf16 (fast + memory
        #           efficient). Only inference/generation is pinned to fp32.
        #
        # Note:     This does NOT call .float() on the model — that would
        #           permanently convert weights to fp32 and break AMP training.
        #           autocast(enabled=False) purely controls the local compute
        #           dtype for this block without modifying any parameter.
        #
        # If changing machines / precision setting:
        #   - This fix is needed with '16-mixed', '16-true', 'bf16-mixed', or
        #     'bf16-true'. It is a no-op with '32-true' (already fp32).
        #   - Keep this block regardless of which AMP mode is used; it is safe.
        # -------------------------------------------------------------------
        if self.lm_type == 'encdec':
            with torch.amp.autocast('cuda', enabled=False):
                outputs = self.language_model.generate(
                    source_input_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    bad_words_ids=bad_words_ids,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                )
        elif self.lm_type == 'dec':
            with torch.amp.autocast('cuda', enabled=False):
                outputs = self.language_model.generate(
                    input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                )
            self.tokenizer.padding_side = 'left'
            # ===== 调试输出 1: Raw Output =====
        # print("\n" + "="*80)
        # print("GENERATE_DIRECT - RAW OUTPUT【334】")
        # print("="*80)
        # print(f"Raw output shape: {outputs.shape}")
        # print(f"Raw output (first sample, all tokens): {outputs[0][0:50].tolist()}")
    
        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)
            # ===== 调试输出 2: Decoded Text =====
        # print(f"\nDecoded output text (first sample):")
        # print(f"{outputs_string[0][0:100]}...")  # Show first 2000 chars
        # print("="*80 + "\n")
        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)
        return outputs_tokens, cleaned_text

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None,
                             do_sample: bool = True):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
            
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        # tasks = [{
                        #     'input':
                        #     ['Generate motion: <Caption_Placeholder>'],
                        #     'output': ['']
                        # }] * len(texts)
                        tasks = [{
                            'input':
                            ['<Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)
                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                    
            elif task == "pred" or task == "m2m":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                # Keep prediction prompt construction centralized in
                # placeholder_fulfill(). Training also enters through that path,
                # so passing the full motion string here prevents inference from
                # using a subtly different head/tail split or boundary token.
                motion_strings = motion_strings_old

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)

            deterministic_prompts = stage == 'test' and not do_sample
            inputs, outputs = self.template_fulfill(
                tasks,
                lengths,
                motion_strings,
                texts,
                stage,
                deterministic=deterministic_prompts)

            # print("outputs:", outputs)
            outputs_tokens, cleaned_text = self.generate_direct(inputs,
                                                                max_length=256,
                                                                num_beams=1,
                                                                do_sample=do_sample,
                                                                temperature=self._sample_temperature,
                                                                top_k=self._sample_top_k,
                                                                top_p=self._sample_top_p)

            return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths)

            # Match training template (template_pretrain.json):
            #   input: "<Motion_Placeholder>"  →  bare motion token string
            #   output: "<Caption_Placeholder>" →  text caption
            if not with_len:
                tasks = [{
                    'input': ['<Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        '<Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            deterministic_prompts = stage == 'test' and not do_sample
            inputs, outputs = self.template_fulfill(
                tasks,
                lengths,
                motion_strings,
                texts,
                stage,
                deterministic=deterministic_prompts)
            # print("inputs[0]:", inputs[0])
            # print("outputs[0]:", outputs[0])
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=do_sample,
                temperature=self._sample_temperature,
                top_k=self._sample_top_k,
                top_p=self._sample_top_p,
            )
            print("outputs_tokens[0]:", outputs_tokens[0])
            print("cleaned_text[0]:", cleaned_text[0])
            return cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            token_list = []
            for s in string_list[1:-1]:
                raw = s.split('_')[-1].replace('>', '')
                try:
                    val = int(raw)
                    # Clamp to valid codebook range
                    val = max(0, min(val, self.m_codebook_size - 1))
                    token_list.append(val)
                except ValueError:
                    # Skip malformed tokens generated by the LM
                    continue
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str, deterministic: bool = False):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')

        # `length` is not always measured in raw frames. During LM pretraining
        # and m2m/pred inference it is already the number of discrete motion
        # tokens, so deriving the split from `length / down_t` can make the
        # prompt shorter than the configured predict_ratio. Count the actual
        # token ids inside the serialized motion string instead.
        #
        # motion_splited has the form:
        #   ["<motion_id_BOS", "<motion_id_a", ..., "<motion_id_EOS", ""]
        # Therefore valid motion token ids occupy indices [1, eos_index).
        num_motion_tokens = max(1, len(motion_splited) - 3)
        predict_tokens = max(1, int(num_motion_tokens * self.predict_ratio))
        masked_head_tokens = max(1, int(num_motion_tokens * self.inbetween_ratio))
        masked_tail_tokens = max(masked_head_tokens,
                                 int(num_motion_tokens *
                                     (1 - self.inbetween_ratio)))

        # Split indices include the initial BOS item at index 0. The prediction
        # input must end with the EOS/separator token because the model sees
        # that exact boundary during training.
        predict_head = 1 + predict_tokens
        masked_head = 1 + masked_head_tokens
        masked_tail = 1 + masked_tail_tokens
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        if not deterministic and random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test',
                         deterministic: bool = False):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            if deterministic:
                input_template = tasks[i]['input'][0]
                output_template = tasks[i]['output'][0]
            else:
                input_template = random.choice(tasks[i]['input'])
                output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i],
                                         deterministic=deterministic))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i],
                                         deterministic=deterministic))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
