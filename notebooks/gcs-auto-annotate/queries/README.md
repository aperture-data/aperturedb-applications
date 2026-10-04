# Queries for the ApertureDB web UI

Paste any of these JSON files into the UI's query editor (or run them from `queries.ipynb`).


## String search

- `01_labels_containing.json` - Case-insensitive substring over label text, then traverse to the posters
- `02_people_by_name.json` - RE2 regular expression over object names
- `03_label_search_boolean.json` - any / not / score combined in one text search
- `04_label_shapes.json` - like and istarts_with patterns
- `05_file_name_search.json` - Prefix search over object names (trigram index)

## Expressions

- `06_happy_frontal_faces.json` - Property-vs-property comparison plus abs() inside the database
- `07_delighted_faces.json` - Arithmetic over properties
- `08_dominant_faces.json` - Posters dominated by one confident face
- `09_sensitive_content.json` - Moderation across two detectors' ranks
- `10_not_yet_enriched.json` - is_missing as a data-quality check

## Graph traversal

- `11_happy_faces_on_tagged_posters.json` - label -> poster -> faces in one request
- `12_label_and_object_together.json` - Posters connected to a label AND an object box
- `13_crowded_posters.json` - Many faces and many objects, from stored counts
- `14_poster_neighbourhood.json` - Everything attached to one poster
- `15_why_flagged.json` - Flagged posters plus the labels that explain them

## Spatial

- `16_overlap_candidates.json` - Posters with both a Logo and a face box
