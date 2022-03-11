<<<<<<< HEAD
# Biweekly progress Report 1 (14.03.22)
=======
# Biweekly progress Report 1 (08.03.22)
>>>>>>> Updating Sanbox
## What did you work on the last two weeks?
### Prodigy
- Setting up prodigy on sandbox
- Getting a very basic "span_manual" recipe to work that displays the facts of the cases and has 3 labels to annotate them: `["Supports judgment", "Neutral", "Opposes verdict"]`
### Organisation
- Set up my own fork of [SwissCourtRulingCorpus](https://github.com/ninabaumgartner/SwissCourtRulingCorpus) on local machine and FDN Sandbox.
- Created Branch for Prodigy setup as seen in prodigy branch from @Skatinger
- Set up my pycharm enviroment, adding the remote Host and the database to it. So that easier access to the files and data are possible.
- Gathering and reading some related literature  on topic
## What new things are working?
- I have access to the sandbox (yay)
- A simple set up of Prodigy (double yay)
- A simple recipe works and gets the correct data from DB (triple yay)
## What are you currently working on?
- Reading prodigy documentation and developing the simple recipes for our specific case
- Figuring out where and how I should save the annotated dataset
- Trying to display the metadata for the law students annotation (we are only displaying the facts section for now)
- Reading more on topic
## What are your current struggles/problems?
- I'm not quite sure how and where the data is saved, maybe I will first try to dump the results in an extern json to not f*** up our beautiful DB ;)
- I'm not quite sure on how to chose the cases (see section below)
## What could we do to help you overcome them?
- Genral Question: Do we only annotate german cases ==> from de?
- I'm still quite unsure on how I should compose my dataset. Currently I'm getting the facts (and the ruling, judgments etc.) with this query:
```sql
SELECT facts FROM de
<<<<<<< HEAD
WHERE NOT (facts like '' ) AND spider like 'CH_BGer' 
=======
WHERE NOT (facts like '' ) AND spider like 'CH_BGer'
>>>>>>> Updating Sanbox
```
Now I have different questions to get a more accurate result:
- We talked about a balanced set: So 50% approval and dismissal, which are definded in the judgment column, correct? 
- Did we already define how many in genreal? So 50% of what?
- Only choose cases from validation and test set ==> How do I know which is which in the scrc?
- Only select judgments where the model has high certainty and is right ==> Where do i see this?
- On the legal area, we talked about penal law which others (aka where was the model second best)? And how do I find these cases in the DB? (Do you have a list of spiders or a certain column where this is defined (I searched the DB and did not find a good answer)?
<<<<<<< HEAD
- Take easy cases aka short cases so should we make a facts limit? 
  - How about 300 word  /1500-1600 character? (halbe/drei viertel Seite)
- Where to dump the results, which column which table? de-> facts? 
=======
- Take easy cases aka short cases so should we make a facts limit?
  - How about 300 word  /1500-1600 character? (halbe/drei viertel Seite)
- Where to dump the results, which column which table? de-> facts?
>>>>>>> Updating Sanbox
