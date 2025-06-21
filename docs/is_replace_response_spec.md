# is_replace_response

* Flag that signals streaming content should replace (not append to) previous content
* Used in AI response streaming to enable edits, corrections, and multi-phase responses (thinking)
* Poe API interface:
```python
class PartialResponse:
    text: str
    is_replace_response: bool = False  # When true, replaces previous content
```
* OpenAI API handling:
  * No native replacement flag in OpenAI API
  * In streaming mode (`stream=True`):
    * Sends normal delta chunks in format with new content:
    ```javascript
    { "choices": [{ "delta": { "content": "new content" } }] }
    ```
    * `content` should fully replace the previous content and can be empty string
    * This requires special handling on the client
  * In non-streaming mode (`stream=False`):
    * Server tracks replacements internally
    * Returns only final content in response