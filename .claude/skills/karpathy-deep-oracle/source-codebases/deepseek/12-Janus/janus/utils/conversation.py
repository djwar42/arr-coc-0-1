# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# <claudes_code_comments>
# ** Function List **
# SeparatorStyle - Enum defining various conversation formatting styles
# Conversation.__init__(...) - Initialize conversation template with roles and separators
# Conversation.get_prompt() - Format conversation history into a single prompt string
# Conversation.get_prompt_for_current_round(content) - Get formatted prompt for current turn
# Conversation.set_system_message(msg) - Update system prompt
# Conversation.append_message(role, msg) - Add new message to conversation history
# Conversation.reset_message() - Clear all messages
# Conversation.update_last_message(msg) - Update most recent assistant response
# Conversation.to_gradio_chatbot() - Convert to Gradio chat widget format
# Conversation.to_openai_api_messages() - Convert to OpenAI API message format
# Conversation.copy() - Create deep copy of conversation state
# register_conv_template(template) - Add conversation template to global registry
# get_conv_template(name) - Retrieve and copy registered conversation template
#
# ** Technical Review **
# This module implements conversation management and formatting for Janus, handling multi-turn
# dialogues with system prompts, user queries, and assistant responses. Adapted from FastChat's
# conversation utilities to support various chat template formats.
#
# **Core Abstraction**:
# Conversation = System Prompt + [(Role, Message), ...] + Formatting Rules → Formatted String
#
# **Supported Separator Styles**:
# 1. **DeepSeek** (Janus default):
#    - Format: "{system}\n\n<|User|>: {query}\n\n<|Assistant|>: {response}<｜end▁of▁sentence｜>"
#    - Two separators: "\n\n" (between turns), "<｜end▁of▁sentence｜>" (end of assistant)
#    - Clean, human-readable format with explicit role markers
#
# 2. **LLAMA2**:
#    - Format: "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{query} [/INST] {response} </s><s>[INST] ..."
#    - Uses special tokens: [INST], [/INST], <<SYS>>, <</SYS>>, </s>
#    - Designed for Llama-2 chat models
#
# 3. **PLAIN**:
#    - Minimal formatting, just messages separated by newlines
#    - Used for simple generation tasks without chat structure
#
# 4. **ALIGNMENT**:
#    - Specialized format for vision-language alignment training
#    - Inserts <image> tags at appropriate positions
#
# **Conversation Flow**:
# 1. Initialize template: `conv = get_conv_template("deepseek")`
# 2. Set system prompt (optional): `conv.set_system_message("You are a helpful assistant.")`
# 3. Add turns: `conv.append_message("User", "Hello")` → `conv.append_message("Assistant", "Hi!")`
# 4. Generate prompt: `prompt = conv.get_prompt()` → "...\n\n<|User|>: Hello\n\n<|Assistant|>: Hi!..."
# 5. Model generates continuation starting from last incomplete turn
#
# **System Prompt Handling**:
# - System message sets task context and behavior guidelines
# - Formatted via system_template (e.g., "{system_message}" or "[INST] <<SYS>>\n{system_message}\n<</SYS>>")
# - Can be empty for models that don't use system prompts
# - Typical content: "You are a helpful language and vision assistant..."
#
# **Roles**:
# - roles tuple defines participant names: ("USER", "ASSISTANT"), ("<|User|>", "<|Assistant|>"), etc.
# - Consistent role names ensure proper formatting across turns
# - Case-sensitive (affects tokenization and stop criteria)
#
# **Separators**:
# - sep: separator after user messages (typically "\n\n" or " ")
# - sep2: separator after assistant messages (typically "</s>" or "<｜end▁of▁sentence｜>")
# - Alternates: User message → sep → Assistant message → sep2 → User message → sep → ...
#
# **Stop Criteria**:
# - stop_str: List of strings that signal end of generation (e.g., ["<|User|>", "<｜end▁of▁sentence｜>"])
# - stop_token_ids: Token IDs that trigger stopping (e.g., [100001] for DeepSeek EOS)
# - Prevents model from continuing into next user turn
#
# **Multi-Turn Support**:
# - Maintains full conversation history in messages list
# - Each turn: [role, message_content]
# - offset parameter: Skip first N messages (for few-shot examples)
# - Last message can be None to prompt model for completion
#
# **Integration with VLChatProcessor**:
# VLChatProcessor uses conversation templates to format input before tokenization:
#   conversations: [{"role": "User", "content": "..."}, ...]
#   → apply_sft_template_for_multi_turn_prompts()
#   → formatted string
#   → tokenizer
#
# **Registered Templates**:
# - "deepseek": Janus default, clean format with <|User|>/<|Assistant|> markers
# - "deepseek_old": Legacy format with "User"/"Assistant" (no brackets)
# - "llava_llama2": LLaVA-style formatting for Llama-2 models
# - "llama-2": Standard Llama-2 chat format
# - "plain": Minimal formatting
# - "alignment": Vision-language alignment training format
#
# **Design Rationale**:
# - **Template abstraction**: Easy to switch between different model chat formats
# - **Stateful conversation**: Maintains history for multi-turn interactions
# - **Copy semantics**: get_conv_template() returns copy to prevent shared state bugs
# - **Format flexibility**: Supports both structured (DeepSeek/Llama) and plain formats
# - **Interoperability**: Conversion to Gradio/OpenAI formats for UI/API integration
#
# **Typical Usage in Janus**:
# ```python
# processor = VLChatProcessor(...)
# conversations = [
#     {"role": "User", "content": "<image_placeholder> What's in this image?"},
#     {"role": "Assistant", "content": ""}  # Empty for generation
# ]
# inputs = processor(conversations=conversations, images=[image])
# ```
#
# **Edge Cases**:
# - Empty system message: Template omits system prompt entirely
# - Incomplete last turn: Model generates completion for partial assistant response
# - Multimodal content: Conversation formatting preserves <image_placeholder> tags
#
# Conversation management ensures prompts are formatted correctly for the language model,
# enabling natural multi-turn dialogues while maintaining proper structure and separators.
# </claudes_code_comments>

"""
From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    DeepSeek = auto()
    PLAIN = auto()
    ALIGNMENT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)

        if self.sep_style == SeparatorStyle.DeepSeek:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if type(message) is tuple:  # multimodal message
                        message, _ = message
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += message + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.ALIGNMENT:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += "<image>\n" + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt_for_current_round(self, content=None):
        """Get current round formatted question prompt during sft training"""
        if self.sep_style == SeparatorStyle.PLAIN:
            formatted_question = "<image>\n"
        elif self.sep_style == SeparatorStyle.DeepSeek:
            formatted_question = (
                f"{self.roles[0]}: " + content.strip() + self.sep + f"{self.roles[1]}:"
            )
        else:
            raise ValueError(f"Unsupported sep_style: {self.sep_style}")
        return formatted_question

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = [{"role": "system", "content": system_prompt}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# llava_llama2 template
register_conv_template(
    Conversation(
        name="llava_llama2",
        system_message="You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language.",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)

# llama2 template
# reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)


# deepseek template
register_conv_template(
    Conversation(
        name="deepseek_old",
        system_template="{system_message}",
        # system_message="You are a helpful assistant. Please answer truthfully and write out your "
        # "thinking step by step to be sure you get the right answer.",
        system_message="",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeek,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["User:", "<｜end▁of▁sentence｜>"],
    )
)
register_conv_template(
    Conversation(
        name="deepseek",
        system_template="{system_message}",
        # system_message="You are a helpful assistant. Please answer truthfully and write out your "
        # "thinking step by step to be sure you get the right answer.",
        system_message="",
        roles=("<|User|>", "<|Assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeek,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["<|User|>", "<｜end▁of▁sentence｜>"]
    )
)

register_conv_template(
    Conversation(
        name="plain",
        system_template="",
        system_message="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PLAIN,
        sep="",
        sep2="",
        stop_token_ids=[2],
        stop_str=["</s>"],
    )
)


register_conv_template(
    Conversation(
        name="alignment",
        system_template="",
        system_message="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ALIGNMENT,
        sep="",
        sep2="",
        stop_token_ids=[2],
        stop_str=["</s>"],
    )
)


if __name__ == "__main__":
    # print("Llama-2 template:")
    # conv = get_conv_template("llama-2")
    # conv.set_system_message("You are a helpful, respectful and honest assistant.")
    # conv.append_message(conv.roles[0], "Hello!")
    # conv.append_message(conv.roles[1], "Hi!")
    # conv.append_message(conv.roles[0], "How are you?")
    # conv.append_message(conv.roles[1], None)
    # print(conv.get_prompt())

    # print("\n")

    print("deepseek template:")
    conv = get_conv_template("deepseek")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi! This is Tony.")
    conv.append_message(conv.roles[0], "Who are you?")
    conv.append_message(conv.roles[1], "I am a helpful assistant.")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
