INSERT INTO "language"(iso_code) VALUES 
	('de'),
	('fr'),
	('it'),
	('en'),
	('na');
INSERT INTO canton(short_code) VALUES 
	('AG'),
	('AI');
INSERT INTO canton_name(canton_id, language_id, "name") VALUES 
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), (SELECT language_id FROM language WHERE iso_code ='de'),'Aargau'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), (SELECT language_id FROM language WHERE iso_code ='de'),'Appenzell Innerrhoden'),
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), (SELECT language_id FROM language WHERE iso_code ='fr'),'Argovie'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), (SELECT language_id FROM language WHERE iso_code ='fr'),'Appenzell Rhodes-Intérieures'),
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), (SELECT language_id FROM language WHERE iso_code ='it'),'Argovia'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), (SELECT language_id FROM language WHERE iso_code ='it'),'Appenzello Interno'),
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), (SELECT language_id FROM language WHERE iso_code ='en'),'Aargau'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), (SELECT language_id FROM language WHERE iso_code ='en'),'Appenzell Innerrhoden'),
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), (SELECT language_id FROM language WHERE iso_code ='na'),'Aargau'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), (SELECT language_id FROM language WHERE iso_code ='na'),'Appenzell Innerrhoden');
INSERT INTO spider("name") VALUES 
	('AG_Weitere'),
	('AI_Aktuell');
INSERT INTO court(canton_id, court_string) VALUES 
	((SELECT canton_id FROM canton WHERE short_code = 'AG'), 'AG_GB'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), 'AI_BZG'),
	((SELECT canton_id FROM canton WHERE short_code = 'AI'), 'AI_KG');
INSERT INTO court_name(court_id, "language_id", "name") VALUES 
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT language_id FROM language WHERE iso_code = 'de'), 'Entscheide Grundbuch und Notariat'),
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT language_id FROM language WHERE iso_code = 'en'), 'Entscheide Grundbuch und Notariat'),
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT language_id FROM language WHERE iso_code = 'fr'), 'Entscheide Grundbuch und Notariat'),
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT language_id FROM language WHERE iso_code = 'it'), 'Entscheide Grundbuch und Notariat'),
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT language_id FROM language WHERE iso_code = 'na'), 'Entscheide Grundbuch und Notariat'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT language_id FROM language WHERE iso_code = 'de'), 'Bezirksgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT language_id FROM language WHERE iso_code = 'en'), 'Bezirksgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT language_id FROM language WHERE iso_code = 'fr'), 'Bezirksgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT language_id FROM language WHERE iso_code = 'it'), 'Bezirksgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT language_id FROM language WHERE iso_code = 'na'), 'Bezirksgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT language_id FROM language WHERE iso_code = 'de'), 'Kantonsgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT language_id FROM language WHERE iso_code = 'en'), 'Kantonsgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT language_id FROM language WHERE iso_code = 'fr'), 'Kantonsgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT language_id FROM language WHERE iso_code = 'it'), 'Kantonsgericht'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT language_id FROM language WHERE iso_code = 'na'), 'Kantonsgericht');
INSERT INTO chamber(court_id, spider_id, chamber_string) VALUES 
	((SELECT court_id FROM court WHERE court_string = 'AG_GB'), (SELECT spider_id FROM spider WHERE name = 'AG_Weitere'), 'AG_GB_001'),
	((SELECT court_id FROM court WHERE court_string = 'AI_BZG'), (SELECT spider_id FROM spider WHERE name = 'AI_Aktuell'), 'AI_BZG_001'),
	((SELECT court_id FROM court WHERE court_string = 'AI_KG'), (SELECT spider_id FROM spider WHERE name = 'AI_Aktuell'), 'AI_KG_001');
INSERT INTO judgment("text") VALUES 
	('approval'),
	('dismissal'),
	('inadmissible'),
	('partial_approval'),
	('partial_dismissal'),
	('unification'),
	('write_off');
INSERT INTO citation_type("name") VALUES 
	('ruling'),
	('law'),
	('commentary');
INSERT INTO section_type("name") VALUES 
	('full_text'),
	('header'),
	('facts'),
	('considerations'),
	('rulings'),
	('footer');
INSERT INTO judicial_person_type("name") VALUES 
	('federal_judge'),
	('deputy_federal_judge'),
	('clerk');
INSERT INTO party_type("name") VALUES 
	('plaintiff'),
	('defendant'),
	('representation_plaintiff'),
	('representation_defendant');
