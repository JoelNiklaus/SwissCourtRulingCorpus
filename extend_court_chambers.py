from collections import OrderedDict
from pathlib import Path
import json

cantons = json.loads(Path("court_chambers.json").read_text())

for canton_key, canton_dict in cantons.items():
    for court_key, court_dict in canton_dict['gerichte'].items():
        for chamber_key, chamber_dict in court_dict['kammern'].items():
            cantons[canton_key]['gerichte'][court_key]['kammern'][chamber_key]['Rechtsgebiete'] = ['']
        cantons[canton_key]['gerichte'][court_key] = OrderedDict(cantons[canton_key]['gerichte'][court_key])
        #cantons[canton_key]['gerichte'][court_key]['Gerichtsinstanz'] = ''
        cantons[canton_key]['gerichte'][court_key].move_to_end('kammern', last=True)

Path("court_chambers_extended.json").write_text(json.dumps(cantons))

