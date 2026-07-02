import json

transcript_path = r"C:\Users\CHRW\.gemini\antigravity-ide\brain\fd769a3a-bab7-4700-8325-3a464720f971\.system_generated\logs\transcript.jsonl"
diffs = []
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            if "output" in str(data):
                content = str(data)
                if "diff --git" in content:
                    diffs.append(line)
        except:
            pass

if diffs:
    print(f"Found {len(diffs)} lines with diffs.")
    with open("scratch/diff_output.txt", "w", encoding="utf-8") as out:
        for d in diffs:
            out.write(d + "\n")
else:
    print("No diffs found.")
