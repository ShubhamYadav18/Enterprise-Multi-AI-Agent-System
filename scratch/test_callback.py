from traces.callbacks import AgentTraceCallback
from langchain_core.outputs import LLMResult, Generation
cb = AgentTraceCallback()
res = LLMResult(
    generations=[[Generation(text="hi")]],
    llm_output={"usage": {"input_tokens": 100, "output_tokens": 50}}
)
cb.on_llm_end(response=res, run_id="123")
print("total:", cb.total_tokens)
print("prompt:", cb.prompt_tokens)
print("completion:", cb.completion_tokens)
