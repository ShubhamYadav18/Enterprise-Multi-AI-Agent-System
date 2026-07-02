from config.settings import get_settings
from langsmith import Client

settings = get_settings()
ls_client = Client()

print("Fetching recent runs...")
# Get the latest root run
project_name = settings.langchain_project
runs = list(ls_client.list_runs(project_name=project_name, is_root=True, limit=1))

if not runs:
    print("No runs found in project:", project_name)
else:
    root_run = runs[0]
    print(f"Found root run: {root_run.id}")
    
    # Now fetch the whole trace
    trace_runs = list(ls_client.list_runs(trace_id=str(root_run.id)))
    print(f"Found {len(trace_runs)} spans in the trace.")
    
    for r in trace_runs:
        if r.run_type == "llm":
            tags = r.tags or []
            print(f"LLM Run: {r.id} | Tags: {tags}")
            print(f"  prompt_tokens: {getattr(r, 'prompt_tokens', 0)}")
            print(f"  completion_tokens: {getattr(r, 'completion_tokens', 0)}")
            print(f"  total_tokens: {getattr(r, 'total_tokens', 0)}")
