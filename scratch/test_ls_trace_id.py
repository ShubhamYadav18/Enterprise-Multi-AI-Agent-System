from config.settings import get_settings
from langsmith import Client

settings = get_settings()
ls_client = Client()

project_name = settings.langchain_project
runs = list(ls_client.list_runs(project_name=project_name, is_root=True, limit=1))
root_run = runs[0]

trace_runs = list(ls_client.list_runs(trace_id=str(root_run.id)))
child_id = None
for r in trace_runs:
    if r.id != root_run.id:
        child_id = r.id
        break

print("Trace ID (root):", root_run.id)
print("Child ID:", child_id)

try:
    test_runs = list(ls_client.list_runs(trace_id=str(child_id)))
    print("Found with child ID:", len(test_runs))
except Exception as e:
    print("Error:", e)
