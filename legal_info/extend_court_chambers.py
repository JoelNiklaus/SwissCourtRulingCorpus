from collections import OrderedDict
from pathlib import Path
import json

cantons = json.loads(
    (Path(__file__).parent / "court_chambers.json").read_text())
current_cantons = json.loads(
    (Path(__file__).parent / "court_chambers_extended.json").read_text())

for canton_key, canton_dict in cantons.items():
    for court_key, court_dict in canton_dict['gerichte'].items():
        for chamber_key, chamber_dict in court_dict['kammern'].items():
            if canton_key in current_cantons and court_key in current_cantons[canton_key]['gerichte'] and chamber_key in current_cantons[canton_key]['gerichte'][court_key]['kammern']:
                cantons[canton_key]['gerichte'][court_key]['kammern'][chamber_key]['Rechtsgebiete'] = current_cantons[
                    canton_key]['gerichte'][court_key]['kammern'][chamber_key]['Rechtsgebiete']
            else:
                cantons[canton_key]['gerichte'][court_key]['kammern'][chamber_key]['Rechtsgebiete'] = [
                    '']
        cantons[canton_key]['gerichte'][court_key] = OrderedDict(
            cantons[canton_key]['gerichte'][court_key])
        #cantons[canton_key]['gerichte'][court_key]['Gerichtsinstanz'] = ''
        cantons[canton_key]['gerichte'][court_key].move_to_end(
            'kammern', last=True)

Path("court_chambers_extended.json").write_text(json.dumps(cantons))
