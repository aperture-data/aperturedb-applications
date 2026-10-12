# Queries for the ApertureDB Web UI

Paste any of these JSON files into the UI's query editor (or run them from `queries.ipynb`).

**One click for the local web UI:** with `docker compose --profile ui up -d` running, open http://localhost:8088/load-queries.html once. It stores all of them in your browser's saved queries (the UI keeps those in the browser, so do this once per browser).


## Headline

- `01_team_sports_movies.json` - Sports label AND three or more people AND family-friendly ratings, in one request
- `02_family_movie_night.json` - A natural-language ask as one query: family-friendly AND animated AND every face smiling
- `03_everyone_looks_happy.json` - Label term AND every face on the poster smiling, in one request

## Discover the Data

- `04_label_vocabulary.json` - Every label Google found, with how many posters carry it (group + count)
- `05_safety_ratings.json` - Content-safety ratings: how many posters per (adult, violent) pair

## Family Friendly

- `06_family_friendly.json` - A label AND both safety ratings VERY_UNLIKELY

## String Search

- `07_labels_containing.json` - Case-insensitive substring over label text, then traverse to the posters
- `08_people_by_name.json` - RE2 regular expression over object names
- `09_label_search_boolean.json` - any / not / score combined in one text search
- `10_label_shapes.json` - like and istarts_with patterns
- `11_label_word_search.json` - Full-text word search over labels with typo tolerance
- `12_file_name_search.json` - Prefix search over object names (trigram index)

## Expressions

- `13_happy_frontal_faces.json` - Property-vs-property comparison plus abs() inside the database
- `14_delighted_faces.json` - Arithmetic over properties
- `15_dominant_faces.json` - Posters dominated by one confident face
- `16_sensitive_content.json` - Moderation across two detectors' ranks
- `17_not_yet_enriched.json` - is_missing as a data-quality check

## Graph Traversal

- `18_happy_faces_on_tagged_posters.json` - label -> poster -> faces in one request
- `19_label_and_object_together.json` - Posters connected to a label AND an object box
- `20_objects_between.json` - Stored-but-below-cut-off objects, via edge properties (no re-ingest to move the threshold)
- `21_crowded_posters.json` - Many faces and many objects, from stored counts
- `22_poster_neighbourhood.json` - Everything attached to one poster
- `23_why_flagged.json` - Flagged posters plus the labels that explain them

## Spatial

- `24_overlap_candidates.json` - Posters with both a Logo and a face box
