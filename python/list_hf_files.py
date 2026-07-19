#!/usr/bin/env python3
import json
import sys
import urllib.request

repository = sys.argv[1]
with urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repository}", timeout=60
) as response:
    metadata = json.load(response)
for item in metadata.get("siblings", []):
    print(item["rfilename"])