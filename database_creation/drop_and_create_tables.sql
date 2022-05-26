/*
  Creation of the database schema v2. 
  Last updated 11 January 2022. If the file has not been updated in a long time, you might want to \d <TABLE> on the postgres database to check whether the information on the tables remained the same.

  CAREFUL! The file contains a (1) DROP command at the top. Comment out if you do not want to DROP the tables.
*/

DROP TABLE IF EXISTS "language", canton, canton_name, spider, court, court_name, chamber, lower_court, "file", decision, 
	judgment, judgment_map, citation_type, citation, section_type, "section", paragraph, num_tokens, file_number,
    judicial_person_type, person, judicial_person, party_type, party;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS "language"(
  language_id SERIAL PRIMARY KEY,
  iso_code CHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS canton(
  canton_id SERIAL PRIMARY KEY,
  short_code text NOT NULL
);

CREATE TABLE IF NOT EXISTS canton_name(
  canton_id INTEGER NOT NULL REFERENCES canton,
  language_id INTEGER NOT NULL REFERENCES "language",
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spider(
  spider_id SERIAL PRIMARY KEY,
  "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS court(
  court_id SERIAL PRIMARY KEY,
  canton_id INTEGER NOT NULL REFERENCES canton,
  court_string TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS court_name(
  court_id INTEGER NOT NULL REFERENCES court,
  language_id INTEGER NOT NULL REFERENCES "language",
  "name" text NOT NULL
);

CREATE TABLE IF NOT EXISTS chamber(
  chamber_id SERIAL PRIMARY KEY,
  court_id INTEGER NOT NULL REFERENCES court,
  spider_id INTEGER NOT NULL REFERENCES spider,
  chamber_string TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS "file"(
  file_id SERIAL PRIMARY KEY,
  file_name TEXT NOT NULL,
  html_url TEXT,
  pdf_url TEXT,
  html_raw TEXT,
  pdf_raw TEXT
);

CREATE TABLE IF NOT EXISTS decision(
  decision_id UUID PRIMARY KEY,
  language_id INTEGER NOT NULL REFERENCES "language",
  chamber_id INTEGER REFERENCES chamber,
  file_id INTEGER NOT NULL REFERENCES "file",
  "date" DATE,
  topic TEXT
);

CREATE TABLE IF NOT EXISTS lower_court(
  lower_court_id SERIAL PRIMARY KEY,
  court_id INTEGER REFERENCES court,
  canton_id INTEGER REFERENCES canton,
  chamber_id INTEGER REFERENCES chamber,
  "date" DATE,
  file_number TEXT,
  decision_id INTEGER NOT NULL REFERENCES decision
);

CREATE TABLE IF NOT EXISTS judgment(
  judgment_id SERIAL PRIMARY KEY,
  "text" TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS judgment_map(
  judgment_id INTEGER REFERENCES judgment,
  decision_id uuid NOT NULL REFERENCES decision ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citation_type(
  citation_type_id SERIAL PRIMARY KEy,
  "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citation(
  citation_id SERIAL PRIMARY KEY,
  citation_type_id INTEGER NOT NULL REFERENCES citation_type,
  decision_id UUID NOT NULL REFERENCES decision ON DELETE CASCADE,
  url TEXT,
  "text" TEXT
);

CREATE TABLE IF NOT EXISTS section_type(
  section_type_id SERIAL PRIMARY Key,
  "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "section"(
  section_id SERIAL PRIMARY KEY,
  decision_id UUID NOT NULL REFERENCES  decision ON DELETE CASCADE,
  section_type_id INTEGER NOT NULL REFERENCES  section_type,
  section_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paragraph(
  paragraph_id SERIAL PRIMARY KEY,
  section_id INTEGER REFERENCES "section",
  decision_id UUID NOT NULL REFERENCES decision ON DELETE CASCADE,
  paragraph_text TEXT NOT NULL,
  first_level INTEGER, 
  second_level INTEGER,
  third_level INTEGER 
);


CREATE TABLE IF NOT EXISTS num_tokens(
  num_tokens_id SERIAL PRIMARY KEY,
  section_id INTEGER NOT NULL REFERENCES "section" ON DELETE CASCADE,
  num_tokens_spacy INTEGER,
  num_tokens_bert INTEGER
);

CREATE TABLE IF NOT EXISTS file_number(
  file_number_id SERIAL PRIMARY KEY,
  decision_id UUID NOT NULL REFERENCES decision ON DELETE CASCADE,
  "text" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judicial_person_type(
  judicial_person_type_id SERIAL PRIMARY KEY,
  "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person(
  person_id SERIAL PRIMARY KEY,
  "name" TEXT,
  is_natural_person BOOLEAN,
  gender TEXT
);

CREATE TABLE IF NOT EXISTS judicial_person(
  decision_id UUID NOT NULL REFERENCES decision ON DELETE CASCADE,
  person_id INTEGER NOT NULL REFERENCES person,
  judicial_person_type_id INTEGER NOT NULL REFERENCES judicial_person_type,
  is_president BOOLEAN
);

CREATE TABLE IF NOT EXISTS party_type(
  party_type_id SERIAL PRIMARY KEY,
  "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS party(
  party_id SERIAL PRIMARY KEY,
  decision_id UUID NOT NULL REFERENCES decision ON DELETE CASCADE,
  person_id INTEGER NOT NULL REFERENCES person,
  party_type_id INTEGER NOT NULL REFERENCES party_type
);