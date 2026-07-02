import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from uuid import uuid4
import time

def test():
    client = Client()
    # Let's get the latest run
    runs = list(client.list_runs(project_name="Enterprise Multi Agent Demo", limit=1))
    if not runs:
        print("No runs found")
        return
    run = runs[0]
    print(f"Run ID: {run.id}")
    print(f"Total tokens: {run.total_tokens}")
    print(f"Prompt tokens: {run.prompt_tokens}")
    print(f"Completion tokens: {run.completion_tokens}")
    
    # Check what is in the run object directly
    print(f"Run attributes: {dir(run)}")
    
if __name__ == "__main__":
    test()
