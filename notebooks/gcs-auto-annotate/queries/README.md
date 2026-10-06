# Queries for the ApertureDB web UI

Paste any of these JSON files into the UI's query editor (or run them from `queries.ipynb`).

**One click for the local web UI:** with `docker compose --profile ui up -d` running, open http://localhost:8088/load-queries.html once. It stores all of them in your browser's saved queries (the UI keeps those in the browser, so do this once per browser).


## Headline

- `01_everyone_looks_happy.json` - Label term AND every face on the poster smiling, in one request

## String search

- `02_labels_containing.json` - Case-insensitive substring over label text, then traverse to the posters
- `03_people_by_name.json` - RE2 regular expression over object names
- `04_label_search_boolean.json` - any / not / score combined in one text search
- `05_label_shapes.json` - like and istarts_with patterns
- `06_file_name_search.json` - Prefix search over object names (trigram index)

## Expressions

- `07_happy_frontal_faces.json` - Property-vs-property comparison plus abs() inside the database
- `08_delighted_faces.json` - Arithmetic over properties
- `09_dominant_faces.json` - Posters dominated by one confident face
- `10_sensitive_content.json` - Moderation across two detectors' ranks
- `11_not_yet_enriched.json` - is_missing as a data-quality check

## Graph traversal

- `12_happy_faces_on_tagged_posters.json` - label -> poster -> faces in one request
- `13_label_and_object_together.json` - Posters connected to a label AND an object box
- `14_crowded_posters.json` - Many faces and many objects, from stored counts
- `15_poster_neighbourhood.json` - Everything attached to one poster
- `16_why_flagged.json` - Flagged posters plus the labels that explain them

## Spatial

- `17_overlap_candidates.json` - Posters with both a Logo and a face box
