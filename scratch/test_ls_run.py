from langsmith import Client
ls_client = Client()
runs = list(ls_client.list_runs(limit=1, run_type="llm"))
r = runs[0]
print(dir(r))
print("total_tokens:", r.total_tokens)
print("prompt_tokens:", getattr(r, "prompt_tokens", None))
print("completion_tokens:", getattr(r, "completion_tokens", None))
print("inputs:", r.inputs)
print("outputs:", r.outputs)
