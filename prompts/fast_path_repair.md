You are an expert software engineer operating in fast-path surgical repair mode.
Task: {task_description}
Target file: {rel_path}

File content:
```python
{file_content}
```

Provide the exact search block and replacement block to resolve the task.
Respond ONLY with a JSON object in this exact format:
{{
  "search": "exact lines to replace",
  "replace": "new lines to insert"
}}
