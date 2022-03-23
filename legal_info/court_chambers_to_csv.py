from pathlib import Path
import json
import pandas as pd

cantons = json.loads(Path("court_chambers.json").read_text())

chambers = []

for canton_key, canton_dict in cantons.items():
    canton = cantons[canton_key]
    for court_key, court_dict in canton_dict['gerichte'].items():
        court = canton['gerichte'][court_key]
        for chamber_key, chamber_dict in court_dict['kammern'].items():
            chamber = court['kammern'][chamber_key]
            chambers.append({
                "canton_key": canton_key,
                "canton_name": canton['de'] if 'de' in canton else "",
                "court_key": court_key,
                "court_name": court['de'] if 'de' in court else "",
                "chamber_key": chamber_key,
                "chamber_name": chamber['de'] if 'de' in chamber else "",
            })
df = pd.DataFrame.from_records(chambers)
df.to_csv("court_chambers.csv", index=False)
