from config.settings import get_settings
settings = get_settings()
print("API Key loaded:", bool(settings.langchain_api_key))
from langsmith import Client
ls_client = Client()
runs = list(ls_client.list_runs(limit=1, run_type="llm"))
print("Runs fetched:", len(runs))
