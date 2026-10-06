# Queries for the ApertureDB web UI

Paste any of these JSON files into the UI's query editor (or run them from `queries.ipynb`).

**One click for the local web UI:** with `docker compose --profile ui up -d` running, open http://localhost:8088/load-queries.html once. It stores all of them in your browser's saved queries (the UI keeps those in the browser, so do this once per browser).


## Headline

- `01_everyone_looks_happy.json` - Label term AND every face on the poster smiling, in one request

## Discover the data

- `02_label_vocabulary.json` - Every label Google found, with how many posters carry it (group + count)
- `03_safety_ratings.json` - Content-safety ratings: how many posters per (adult, violent) pair

## Content safety

- `04_not_safe_for_work.json` - Posters rated LIKELY or VERY_LIKELY adult
- `05_not_safe_for_work_labels.json` - What those posters are about: labels grouped over exactly that set
- `06_family_friendly.json` - The opposite filter: a label AND both ratings VERY_UNLIKELY

## String search

- `07_labels_containing.json` - Case-insensitive substring over label text, then traverse to the posters
- `08_people_by_name.json` - RE2 regular expression over object names
- `09_label_search_boolean.json` - any / not / score combined in one text search
- `10_label_shapes.json` - like and istarts_with patterns
- `11_file_name_search.json` - Prefix search over object names (trigram index)

## Expressions

- `12_happy_frontal_faces.json` - Property-vs-property comparison plus abs() inside the database
- `13_delighted_faces.json` - Arithmetic over properties
- `14_dominant_faces.json` - Posters dominated by one confident face
- `15_sensitive_content.json` - Moderation across two detectors' ranks
- `16_not_yet_enriched.json` - is_missing as a data-quality check

## Graph traversal

- `17_happy_faces_on_tagged_posters.json` - label -> poster -> faces in one request
- `18_label_and_object_together.json` - Posters connected to a label AND an object box
- `19_crowded_posters.json` - Many faces and many objects, from stored counts
- `20_poster_neighbourhood.json` - Everything attached to one poster
- `21_why_flagged.json` - Flagged posters plus the labels that explain them

## Spatial

- `22_overlap_candidates.json` - Posters with both a Logo and a face box
