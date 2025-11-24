# Conversation Formatter

**Category**: Codebase
**File**: `ovis/model/conversation_formatter.py`

## Purpose

Format conversations into prompts with proper chat template.

## Classes

### ConversationFormatter (Abstract)

Base class defining interface:
- `format_conversation()`: Convert messages to prompt
- `format_query()`: Format single query

### Qwen3ConversationFormatter

Qwen3-specific chat template implementation.

**Template Format**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Question?<|im_end|>
<|im_start|>assistant
```

## Usage

```python
formatter = Qwen3ConversationFormatter()
prompt = formatter.format_conversation(messages)
```
