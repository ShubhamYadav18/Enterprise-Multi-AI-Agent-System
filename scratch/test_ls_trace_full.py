from config.settings import get_settings
from langsmith import Client

settings = get_settings()
ls_client = Client()

project_name = settings.langchain_project
runs = list(ls_client.list_runs(project_name=project_name, is_root=True, limit=1))
root_run = runs[0]

trace_runs = list(ls_client.list_runs(trace_id=str(root_run.id)))

print(f"Total spans: {len(trace_runs)}")
for r in trace_runs:
    print(f"ID: {r.id} | Type: {r.run_type} | Name: {r.name} | Parent: {r.parent_run_id} | Tags: {r.tags}")
