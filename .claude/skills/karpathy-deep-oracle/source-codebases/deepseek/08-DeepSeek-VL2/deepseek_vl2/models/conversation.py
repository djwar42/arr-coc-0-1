"""
From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import IntEnum, auto
from typing import Any, Dict, List

# <claudes_code_comments>
# ** Function List **
# SeparatorStyle - IntEnum: DeepSeek(1), DeepSeekV2(2), PLAIN(3), ALIGNMENT(4)
# Conversation.__init__() - Dataclass with name, roles, messages, sep_style, separators
# Conversation.get_prompt() - Formats messages → full text prompt (core method)
# Conversation.set_system_message(system_message) - Updates self.system_message
# Conversation.append_message(role, message) - Appends [role, message] to self.messages
# Conversation.update_last_message(message) - Sets self.messages[-1][1] = message
# Conversation.reset_message() - Clears self.messages = []
# Conversation.to_gradio_chatbot() - Converts to [[user, bot], ...] format
# Conversation.to_openai_api_messages() - Converts to [{"role": "user", "content": ...}, ...]
# Conversation.copy() - Deep copy with new list instances
# Conversation.dict() - Serializes to dict (template_name, system_message, roles, messages, offset)
# register_conv_template(template, override) - Adds template to conv_templates dict
# get_conv_template(name) - Returns conv_templates[name].copy()
#
# ** Technical Review **
# Conversation management for DeepSeek-VL2 chat interface, handling multi-turn dialogues
# with proper role formatting and special token insertion. Adapted from FastChat library.
#
# SEPARATORSTYLE ENUM (detailed formats):
# 1. DeepSeek (original format):
#    - Role markers: "User:", "Assistant:"
#    - User separator: "\n\n" (double newline)
#    - Assistant separator: "<｜end▁of▁sentence｜>" (EOS token)
#    - Example: "User: Hello\n\nAssistant: Hi!<｜end▁of▁sentence｜>User: How are you?\n\nAssistant: "
#
# 2. DeepSeekV2 (SFT format with special markers):
#    - User input wrapped: "<｜sft▁begin｜>\n{message}\n<｜sft▁end｜>"
#    - Assistant response: "{message}<｜end▁of▁sentence｜>"
#    - Example: "<｜sft▁begin｜>\nHello\n<｜sft▁end｜>Hi!<｜end▁of▁sentence｜>"
#    - Used for supervised fine-tuning (SFT) data format
#
# 3. PLAIN (minimal format):
#    - No role markers, no special tokens
#    - Messages separated by sep/sep2 only
#    - Used for simple text completion tasks
#    - Example: "Hello\n\nHi!\n\n"
#
# 4. ALIGNMENT (vision-specific format):
#    - Forces "<image>\n" before even-indexed messages
#    - Used for vision-language alignment training
#    - Example: "<image>\n\nResponse text\n\n"
#
# CONVERSATION DATACLASS (field details):
# - name: str
#   Template identifier ("deepseek", "deepseekv2", "plain", "alignment")
#
# - system_template: str = "{system_message}"
#   Template string for system prompt insertion
#   Can be customized: "System: {system_message}\n\n"
#
# - system_message: str = ""
#   Actual system prompt content
#   Default empty (no system prompt)
#   Example: "You are a helpful assistant. Answer concisely."
#
# - roles: List[str] = (("USER", "ASSISTANT"),)
#   Tuple of (user_role, assistant_role) names
#   Examples:
#     - ("<|User|>", "<|Assistant|>") for DeepSeek
#     - ("|<User>|", "|<Assistant>|") for DeepSeekV2 variant
#     - ("", "") for PLAIN style (no role markers)
#
# - messages: List[List[str]] = ()
#   Conversation history as [[role, content], ...]
#   Example: [["<|User|>", "Hello"], ["<|Assistant|>", "Hi!"], ["<|User|>", "How are you?"], ["<|Assistant|>", None]]
#   None content: Indicates incomplete turn (waiting for assistant response)
#
# - offset: int = 0
#   Number of initial messages to skip (for few-shot examples)
#   Used in to_gradio_chatbot() and to_openai_api_messages()
#
# - sep_style: SeparatorStyle = SeparatorStyle.DeepSeek
#   Determines formatting logic in get_prompt()
#
# - sep: str = "\n"
#   Primary separator (typically between user messages)
#   DeepSeek: "\n\n", DeepSeekV2: "\n<｜sft▁end｜>", PLAIN: ""
#
# - sep2: str = None
#   Secondary separator (typically after assistant messages)
#   DeepSeek: "<｜end▁of▁sentence｜>", DeepSeekV2: "<｜end▁of▁sentence｜>"
#
# - stop_str: str = None
#   String patterns that trigger generation stop
#   Example: ["User:", "<｜end▁of▁sentence｜>"]
#   Used by generation code to detect end of turn
#
# - stop_token_ids: List[int] = None
#   Token IDs that trigger generation stop
#   Example: [100001] (EOS token ID)
#   More reliable than string matching
#
# PROMPT FORMATTING (get_prompt detailed algorithm):
# Algorithm for SeparatorStyle.DeepSeek:
#
# Input:
#   system_message = ""
#   roles = ("<|User|>", "<|Assistant|>")
#   messages = [
#     ["<|User|>", "Hello"],
#     ["<|Assistant|>", "Hi!"],
#     ["<|User|>", "How are you?"],
#     ["<|Assistant|>", None]
#   ]
#   sep = "\n\n"
#   sep2 = "<｜end▁of▁sentence｜>"
#
# Algorithm:
#   1. Format system prompt:
#      system_prompt = system_template.format(system_message="")
#      system_prompt = "" (empty, skipped)
#      ret = ""
#
#   2. Iterate messages with alternating separators:
#      i=0: role="<|User|>", message="Hello"
#           ret += "<|User|>: Hello" + sep
#           ret = "<|User|>: Hello\n\n"
#
#      i=1: role="<|Assistant|>", message="Hi!"
#           ret += "<|Assistant|>: Hi!" + sep2
#           ret = "<|User|>: Hello\n\n<|Assistant|>: Hi!<｜end▁of▁sentence｜>"
#
#      i=2: role="<|User|>", message="How are you?"
#           ret += "<|User|>: How are you?" + sep
#           ret = "<|User|>: Hello\n\n<|Assistant|>: Hi!<｜end▁of▁sentence｜><|User|>: How are you?\n\n"
#
#      i=3: role="<|Assistant|>", message=None
#           ret += "<|Assistant|>:"
#           ret = "<|User|>: Hello\n\n<|Assistant|>: Hi!<｜end▁of▁sentence｜><|User|>: How are you?\n\n<|Assistant|>:"
#
# Output:
#   "<|User|>: Hello\n\n<|Assistant|>: Hi!<｜end▁of▁sentence｜><|User|>: How are you?\n\n<|Assistant|>:"
#
# Why None for last message?
#   - Leaves prompt ready for model to complete
#   - Role marker present, but no content yet
#   - Model generates response after "<|Assistant|>:" prompt
#
# Algorithm for SeparatorStyle.DeepSeekV2:
#
# Input:
#   roles = ("|<User>|", "|<Assistant>|")
#   messages = [["User", "Hello"], ["Assistant", "Hi!"], ["User", "How are you?"], ["Assistant", None]]
#   sep = "\n<｜sft▁end｜>"
#   sep2 = "<｜end▁of▁sentence｜>"
#
# Algorithm:
#   1. System prompt (same as DeepSeek)
#
#   2. Iterate messages with SFT markers:
#      i=0: role="User", message="Hello"
#           ret += "<｜sft▁begin｜>\n" + "Hello" + sep
#           ret = "<｜sft▁begin｜>\nHello\n<｜sft▁end｜>"
#
#      i=1: role="Assistant", message="Hi!"
#           ret += "Hi!" + sep2
#           ret = "<｜sft▁begin｜>\nHello\n<｜sft▁end｜>Hi!<｜end▁of▁sentence｜>"
#
#      i=2: role="User", message="How are you?"
#           ret += "<｜sft▁begin｜>\n" + "How are you?" + sep
#           ret = "<｜sft▁begin｜>\nHello\n<｜sft▁end｜>Hi!<｜end▁of▁sentence｜><｜sft▁begin｜>\nHow are you?\n<｜sft▁end｜>"
#
#      i=3: role="Assistant", message=None
#           ret unchanged (empty message, no addition)
#
# Output:
#   "<｜sft▁begin｜>\nHello\n<｜sft▁end｜>Hi!<｜end▁of▁sentence｜><｜sft▁begin｜>\nHow are you?\n<｜sft▁end｜>"
#
# Note: DeepSeekV2 format omits role markers entirely, uses SFT wrappers instead
#
# IMAGE TOKEN HANDLING (detailed integration):
# <image> tokens preserved verbatim in message content:
#
# Input messages:
#   messages = [
#     ["<|User|>", "<image>\nWhat is in this image?"],
#     ["<|Assistant|>", None]
#   ]
#
# get_prompt() output:
#   "<|User|>: <image>\nWhat is in this image?\n\n<|Assistant|>:"
#
# Processing flow:
#   1. Conversation.get_prompt() → text with <image> placeholders
#   2. DeepseekVLV2Processor.tokenize_with_images() → splits text by <image>
#   3. Processor inserts actual image tokens (313-2509 tokens per image)
#   4. Final token sequence: [BOS, "<|User|>:", image_tokens×757, "What", "is", ..., "\n\n", "<|Assistant|>:"]
#
# Multi-image support:
#   messages = [["<|User|>", "<image><image> Compare these images."], ...]
#   → Two separate <image> tokens in text
#   → Processor replaces each with corresponding image tile embeddings
#
# SPECIAL TOKENS (detailed usage):
# 1. <｜end▁of▁sentence｜> (EOS):
#    - Unicode character: ｜ (U+FF5C FULLWIDTH VERTICAL LINE)
#    - Purpose: Marks end of assistant turn
#    - stop_token_ids: [100001] typically
#    - Generation stops when this token is produced
#    - Why fullwidth? Avoids confusion with regular | character
#
# 2. <｜sft▁begin｜> / <｜sft▁end｜>:
#    - Wraps user input in DeepSeekV2 format
#    - Purpose: Clearly demarcates training data boundaries
#    - Helps model distinguish user input vs assistant response
#    - Used in supervised fine-tuning (SFT) dataset creation
#
# 3. <|User|> / <|Assistant|>:
#    - Role markers in DeepSeek format
#    - Added to tokenizer vocabulary as special tokens
#    - Prevent splitting: "<|User|>" treated as single token
#    - Model learns to recognize role transitions
#
# 4. <image>:
#    - Vision embedding placeholder
#    - NOT replaced during conversation formatting
#    - Replaced during tokenization by processor
#    - Expands to 313-2509 tokens depending on image resolution
#
# 5. Grounding tokens (added by processor, not conversation):
#    - <|ref|>...<|/ref|>: Referring expression boundaries
#    - <|det|>...<|/det|>: Detection output boundaries
#    - <|grounding|>: Enables grounded captioning mode
#    - Example: "<|ref|>red car<|/ref|>" → model localizes "red car"
#
# TEMPLATE REGISTRATION (global registry pattern):
# conv_templates: Dict[str, Conversation] = {}
#
# register_conv_template(template, override=False):
#   - Stores template in global dict
#   - override=False: Raises error if name already exists
#   - override=True: Replaces existing template
#   - Enables centralized template management
#
# get_conv_template(name):
#   - Returns conv_templates[name].copy()
#   - copy() creates fresh instance to prevent mutation
#   - Each retrieval gets independent conversation state
#   - Critical for concurrent chat sessions
#
# Registered templates (lines 301-381):
#   1. "deepseek" - Standard DeepSeek format with <|User|>/<|Assistant|>
#   2. "deepseekv2" - SFT format with |<User>|/|<Assistant>| (note position swap)
#   3. "plain" - Minimal format, no role markers
#   4. "alignment" - Vision alignment training format
#
# GRADIO INTEGRATION (to_gradio_chatbot detailed):
# Gradio Chatbot component expects: List[List[str, str]]
# Format: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
#
# Conversion algorithm:
# Input:
#   messages = [
#     ["<|User|>", "Hello"],
#     ["<|Assistant|>", "Hi!"],
#     ["<|User|>", "How are you?"],
#     ["<|Assistant|>", "I'm good!"]
#   ]
#   offset = 0
#
# Algorithm:
#   ret = []
#   i=0 (even): ret.append(["Hello", None]) → ret = [["Hello", None]]
#   i=1 (odd):  ret[-1][-1] = "Hi!" → ret = [["Hello", "Hi!"]]
#   i=2 (even): ret.append(["How are you?", None]) → ret = [["Hello", "Hi!"], ["How are you?", None]]
#   i=3 (odd):  ret[-1][-1] = "I'm good!" → ret = [["Hello", "Hi!"], ["How are you?", "I'm good!"]]
#
# Output:
#   [["Hello", "Hi!"], ["How are you?", "I'm good!"]]
#
# Rendering in Gradio:
#   User: Hello
#   Bot: Hi!
#   User: How are you?
#   Bot: I'm good!
#
# Handling incomplete turns:
#   If last message is [user, None], it appears as [user_msg, None] in list
#   Gradio renders as user message with no bot response yet
#
# OPENAI API COMPATIBILITY (to_openai_api_messages detailed):
# OpenAI Chat Completion format:
#   [
#     {"role": "system", "content": "System prompt"},
#     {"role": "user", "content": "User message"},
#     {"role": "assistant", "content": "Bot response"},
#     ...
#   ]
#
# Conversion algorithm:
# Input:
#   system_message = "You are helpful."
#   messages = [["<|User|>", "Hello"], ["<|Assistant|>", "Hi!"]]
#   offset = 0
#
# Algorithm:
#   1. Add system message:
#      ret = [{"role": "system", "content": "You are helpful."}]
#
#   2. Convert messages (ignore role names, use position):
#      i=0 (even): ret.append({"role": "user", "content": "Hello"})
#      i=1 (odd):  ret.append({"role": "assistant", "content": "Hi!"})
#
# Output:
#   [
#     {"role": "system", "content": "You are helpful."},
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hi!"}
#   ]
#
# Note: Discards DeepSeek-specific role markers, uses position parity
#
# COPY AND SERIALIZATION (state management):
# copy() creates deep copy:
#   - All string fields copied by value (immutable)
#   - messages: [[x, y] for x, y in self.messages] creates new list instances
#   - Critical: Each list element is also new [x, y] (not just shallow copy)
#   - Prevents mutations from affecting other conversation instances
#
# dict() serializes to JSON-friendly format:
#   {
#     "template_name": "deepseek",
#     "system_message": "",
#     "roles": ("<|User|>", "<|Assistant|>"),
#     "messages": [["<|User|>", "Hello"], ["<|Assistant|>", "Hi!"]],
#     "offset": 0
#   }
#
# Use cases:
#   - Save conversation to database/file
#   - Load conversation from checkpoint
#   - Transfer conversation between sessions
#   - Log conversation history for analysis
#
# DESIGN PHILOSOPHY:
# 1. Flexible: Multiple formats via SeparatorStyle enum
#    - Easy to add new formats (just add enum value + logic in get_prompt)
#    - Each format optimized for specific use case (chat, SFT, alignment)
#
# 2. Extensible: Template registry pattern
#    - New templates registered at module load time
#    - Can override templates at runtime if needed
#    - Easy to experiment with prompt formats
#
# 3. Compatible: Export to multiple formats
#    - Gradio: Web UI integration
#    - OpenAI: API compatibility for standard clients
#    - Can add more formats as needed (Anthropic, Cohere, etc.)
#
# 4. Stateful: Maintains conversation history
#    - append_message() adds to history
#    - update_last_message() modifies in-place
#    - reset_message() clears for new conversation
#    - Supports multi-turn dialogue without external state management
#
# INTEGRATION WITH DEEPSEEK-VL2 PIPELINE:
# Full workflow:
#
# 1. User creates conversation:
#    conv = get_conv_template("deepseek")
#    conv.append_message(conv.roles[0], "<image>\nDescribe this image.")
#    conv.append_message(conv.roles[1], None)
#
# 2. Format to text prompt:
#    prompt = conv.get_prompt()
#    # → "<|User|>: <image>\nDescribe this image.\n\n<|Assistant|>:"
#
# 3. Processor tokenizes:
#    prepare = processor(prompt=prompt, images=[pil_image])
#    # → VLChatProcessorOutput with input_ids, images, masks
#
# 4. Model generates:
#    outputs = model.generate(**prepare)
#    response = processor.decode(outputs[0])
#
# 5. Update conversation:
#    conv.update_last_message(response)
#
# 6. Continue multi-turn:
#    conv.append_message(conv.roles[0], "What else can you tell me?")
#    conv.append_message(conv.roles[1], None)
#    # Repeat steps 2-5
#
# Why separate conversation from processor?
# - Conversation: High-level dialogue management (roles, turns, formatting)
# - Processor: Low-level tokenization and image processing
# - Separation of concerns: Easier to modify each independently
# - Conversation can be used standalone for text-only chat
# - Processor can handle images without knowing conversation structure
# </claudes_code_comments>


class SeparatorStyle(IntEnum):
    """Separator styles."""

    DeepSeek = auto()
    DeepSeekV2 = auto()
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
    sep_style: SeparatorStyle = SeparatorStyle.DeepSeek
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
        elif self.sep_style == SeparatorStyle.DeepSeekV2:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if role == "User":
                        ret += "<｜sft▁begin｜>\n" + message + self.sep #<｜sft▁begin｜>User Input<｜sft▁end｜>\nResponse<｜end▁of▁sentence｜>
                    else:
                        ret += message + self.sep2
                else:
                    ret = ret
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
                        ret += '<image>\n' + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

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
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# register_conv_template(
#     Conversation(
#         name="deepseek",
#         system_template="{system_message}",
#         # system_message="You are a helpful assistant. Please answer truthfully and write out your "
#         # "thinking step by step to be sure you get the right answer.",
#         system_message="",
#         roles=("User", "Assistant"),
#         messages=(),
#         offset=0,
#         sep_style=SeparatorStyle.DeepSeek,
#         sep="\n\n",
#         sep2="<｜end▁of▁sentence｜>",
#         stop_token_ids=[100001],
#         stop_str=["User:", "<｜end▁of▁sentence｜>"]
#     )
# )
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
        stop_str=["User:", "<｜end▁of▁sentence｜>"]
    )
)
# register_conv_template(
#     Conversation(
#         name="deepseekv2",
#         system_template="{system_message}",
#         system_message="",
#         roles=("User", "Assistant"),
#         messages=(),
#         offset=0,
#         sep_style=SeparatorStyle.DeepSeekV2,
#         sep="\n<｜sft▁end｜>",
#         sep2="<｜end▁of▁sentence｜>",
#         stop_token_ids=[100001],
#         stop_str=["User:", "<｜end▁of▁sentence｜>"]
#     )
# )
register_conv_template(
    Conversation(
        name="deepseekv2",
        system_template="{system_message}",
        system_message="",
        roles=("|<User>|", "|<Assistant>|"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeekV2,
        sep="\n<｜sft▁end｜>",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["User:", "<｜end▁of▁sentence｜>"]
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
        stop_token_ids=[100001],
        stop_str=['</s>'],
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
        stop_token_ids=[100001],
        stop_str=['</s>'],
    )
)


if __name__ == "__main__":
    print("deepseek template:")
    conv = get_conv_template("deepseek")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi! This is Tony.")
    conv.append_message(conv.roles[0], "Who are you?")
    conv.append_message(conv.roles[1], "I am a helpful assistant.")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("deepseekv2 template:")
    conv = get_conv_template("deepseekv2")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi! This is Tony.")
    conv.append_message(conv.roles[0], "Who are you?")
    conv.append_message(conv.roles[1], "I am a helpful assistant.")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
