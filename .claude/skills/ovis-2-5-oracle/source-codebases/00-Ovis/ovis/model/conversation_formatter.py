"""Conversation Formatter - Chat Template Management for Ovis

Handles conversation formatting for different LLM backends with special token
handling for images and videos.
"""

# <claudes_code_comments>
# ** Function List **
# ConversationFormatter.__init__ - initialize with tokenizer and special tokens
# ConversationFormatter._tokenize_with_image_symbol - tokenize text with <image>/<video> placeholders
# ConversationFormatter.format - abstract method to format conversation with labels
# ConversationFormatter.format_query - abstract method to format single query
# Qwen3ConversationFormatter.__init__ - initialize Qwen3-specific formatter with role mappings
# Qwen3ConversationFormatter._initialize_gpt_token_nums - count tokens for thinking/no-thinking prefixes
# Qwen3ConversationFormatter.format - format conversation with Qwen3 chat template and labels
# Qwen3ConversationFormatter.format_query - format single query for inference
#
# ** Technical Review **
# This module provides conversation formatting with chat templates for different LLM backends.
# ConversationFormatter is abstract base class defining interface for formatting conversations
# into tokenized sequences with proper labels for training.
#
# Key method: _tokenize_with_image_symbol() handles special tokens <image> and <video>:
# - Splits text by token symbol
# - Tokenizes each chunk separately
# - Inserts IMAGE_TOKEN_ID or VIDEO_TOKEN_ID between chunks
# - Returns flat list of token IDs preserving special token positions
#
# Qwen3ConversationFormatter implements Qwen3's chat template:
# Format: <|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n
#
# Role mapping (from2role dict):
# - "human" → "<|im_start|>user\n"
# - "gpt" → "<|im_start|>assistant\n"
# - "system" → "<|im_start|>system\n"
# - "ignored_gpt" → assistant role but labels masked
#
# Thinking mode handling:
# - Detects <think>...</think> tags in assistant messages
# - If thinking present: no empty_think prefix added
# - If no thinking: adds empty_think prefix ("<think>\n\n</think>\n\n") to maintain format
# - _initialize_gpt_token_nums() counts tokens in "assistant" + empty_think for label alignment
#
# Label generation (format() method):
# - User messages: ALL tokens labeled IGNORE_ID (not learned)
# - Assistant messages: Tokens after role prefix learned, last \n ignored
# - Label alignment uses gpt_token_num to skip role prefix + empty_think
# - label_ids[gpt_token_num:-1] = token_ids[gpt_token_num:-1] learns assistant response only
#
# Training flow: Conversations → format() → (prompt, input_ids, labels)
# Labels ensure model only learns to generate assistant responses, not user queries or role markers
#
# Inference flow: Query → format_query() → (prompt, input_ids) with generation_preface
# generation_preface allows forcing start of assistant response (e.g., for thinking mode)
#
# Special handling:
# - Removes trailing \n from input_ids after final <|im_end|> for clean generation start
# - Preserves all special tokens (<image>, <video>) in tokenized sequence
# - Maintains exact token alignment between input_ids and labels
# </claudes_code_comments>

import copy
from abc import ABC, abstractmethod
from typing import List, Dict

from ovis.util.constants import IMAGE_TOKEN_ID, IGNORE_ID, IMAGE_TOKEN, VIDEO_TOKEN_ID, VIDEO_TOKEN


class ConversationFormatter(ABC):
    support_tokenizer_types = None

    def __init__(self, tokenizer):
        tokenizer_type = type(tokenizer).__name__
        assert tokenizer_type in self.support_tokenizer_types, \
            f'Invalid tokenizer type, expected one from `{self.support_tokenizer_types}`, but got `{tokenizer_type}`'
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID
        self.im_end = None
        self.video_token = VIDEO_TOKEN
        self.video_token_id = VIDEO_TOKEN_ID

    def _tokenize_with_image_symbol(self, text):
        if text.find(self.video_token) != -1:
            token = self.video_token
            token_id = self.video_token_id
        else:
            token = self.image_token
            token_id = self.image_token_id

        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(token)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(token_id)
        return token_ids

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None, enable_thinking=False):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass

class Qwen3ConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['QWenTokenizer', 'Qwen2TokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.from2role = {
            "system": "<|im_start|>system\n",
            "human": "<|im_start|>user\n",
            "gpt": "<|im_start|>assistant\n",
            "ignored_gpt": "<|im_start|>assistant\n",
        }
        
        self.im_end = "<|im_end|>\n"
        self.empty_think = "<think>\n\n</think>\n\n"
        self.gpt_token_nums = None

    def _initialize_gpt_token_nums(self) -> Dict[str, int]:
        think_prefix = self.from2role["gpt"]
        think_num = len(
            self.tokenizer(think_prefix, add_special_tokens=False).input_ids
        )
        no_think_prefix = self.from2role["gpt"] + self.empty_think
        no_think_num = len(
            self.tokenizer(no_think_prefix, add_special_tokens=False).input_ids
        )
        return {'think': think_num, 'no_think': no_think_num}

    # enable_thinking is deprecated
    def format(self, conversations: List[Dict], generation_preface=None, enable_thinking=False):
        conversations = copy.deepcopy(conversations)

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = ""
        input_ids = []
        labels = []
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"]
            has_thinking = '<think>' in message and '</think>' in message
            if frm == 'gpt' and not has_thinking and generation_preface is None:
                text = role + self.empty_think + message
            else:
                text = role + message
            
            if self.gpt_token_nums is None:
                self.gpt_token_nums = self._initialize_gpt_token_nums()
            gpt_token_num = self.gpt_token_nums['think'] if has_thinking else self.gpt_token_nums['no_think']
            
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt" and generation_preface is None:
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[gpt_token_num:-1] = token_ids[gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        if conversations[-1]['from'] == "gpt" and generation_preface is None:
            # remove the last `\n` following `im_end` in input_ids
            input_ids.pop()
            labels.pop()

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface="", enable_thinking=False):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface, enable_thinking=enable_thinking)

        return prompt, input_ids
