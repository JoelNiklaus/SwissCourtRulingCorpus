from spacy import language
import prodigy
import psycopg2
from psycopg2 import sql
import os

## Docs
#
# This recipe is used to load judgments directly from the scrc database configured in the environment variables
# The recipe servers the data to the prodigy app, allowing to label the judgments by hand.
#

# a generator which returns records from the database, using batched fetching
# format is for the `choice` view of the prodigy app
def db_stream(language, spider):
  # scrc database connection configuration string
  config = "dbname=scrc user={} password={} host=localhost port=5432".format(os.environ["DB_USER"], os.environ["DB_PASSWORD"])

  # prodigy database connection configuration string, to fetch processed ids
  # TODO:
  # - remove hardcoded login credentials
  # - find the ids which have been labeled already
  # - check which records have already been labeled
  # last two are easier done once the annotations are direclty fed to the scrc database, allowing for a fast query
  # instead of parsing all the stored data from prodigy
  latest_id = 0
  ids_done = []

  # postgres is much faster parsing VALUES() to relations than using a NOT IN query
  # use sql.Identifier and parameters to prevent sql injections
  query = sql.SQL("SELECT id, spider, rulings " \
          "FROM {} " \
          "WHERE spider = %s " \
          "AND id > %s ").format(sql.Identifier(language))
  params = (spider, latest_id)
  # if any judgments have been labeled already, filter them out
  if len(ids_done) > 0:
    ids_done = ", ".join(["({})".format(id) for id in ids_done]) # format for sql query
    query += " AND id NOT IN (VALUES(%s);"
    params += (ids_done,)
  
  # TODO: ensure the connection is closed and reopened on every generator call in case problems with the connection arise
  with psycopg2.connect(config) as connection:
    with connection.cursor() as cursor:
      cursor.execute(query, params)
<<<<<<< HEAD
      #print(cursor.fetchone())
=======
>>>>>>> Added prodigy set up from @Skatinger to ./prodigy.
      while True:
        rows = cursor.fetchmany(1000)
        if not rows:
          break
        for row in rows:
          idx, spider, ruling = row
          res = { "id": idx, "text": ruling }

          if len(res) > 0:
              yield res

# helper function for recipe command line input
def split_string(string):
  return string.split(",")

# the recipe itself, for info on how to use it, see the prodigy docs
@prodigy.recipe(
  "judgment-outcome",
  spider=("Name of dataset (spider)", 'positional', None, str),
  language=("The language to use for the recipe", 'positional', None, str)
)

# function called by the @prodigy-recipe definition
def judgment_outcome(spider: str, language: str):

  # define the default labels (from the enum `Judgmenet` defined in scrc)
  labels = ['approval', 'partial_approval', 'dismissal', 'partial_dismissal', 'inadmissible', 'write_off', 'unification']

  # define stream (generator)
  stream = db_stream(language, spider)
  stream = add_options(stream, labels)

  return {
    "dataset": spider,
    "view_id": "choice",
    "stream": stream,
    "config": {
      "choice_style": "single",
      "choice_auto_accept:": "true"
    }
  }

# small helper to add the label options to the stream
# will later be displayed as choices in the frontend
def add_options(stream, options):
  options = [{"id": opt, "text": opt} for opt in options]
  for task in stream:
    task["options"] = options
    yield task
