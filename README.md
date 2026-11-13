# aperturedb-applications
Public repo for demos and applications using ApertureDB

## Versioning

The root `VERSION` file contains the release version in `MAJOR.MINOR.PATCH`
format. Bump it before merging a new docs-content release into `main`.
On `main`, the docs-content workflow creates the matching Git tag, publishes
an image tagged `<VERSION>`, and sends that same tag to infra to update the
develop component pin. Reruns reuse a Git tag only when it points to the same
commit; an existing tag on another commit requires a version bump.
The release version is embedded in the image for GitHub source and Colab links.
Develop and pull request builds validate the image without publishing it.

## Server compatibility

These examples require ApertureDB server 1.0 and Python SDK `aperturedb>=1.0,<2`.
Media additions use explicit application classes. Descriptor search spaces use
descriptor classes with native vector indexes.

## Existing data and generated files

These applications target server 1.0. Updating code does not rename stored
application classes. Inspect `GetSchema` (`nodes` and `edges`) before reusing an
existing database: converted built-in categories have names such as `Image`,
`Video` and `Blob`, while these examples use explicit application classes. On a
dedicated application database, use the native `RenameClass` command to align
those categories with the configured names, or reingest through the updated
loaders. Shared databases need an application-specific export/reingestion plan.
Regenerate connection CSVs and cached identifiers against the native schema;
old `_Image@...`/`_Descriptor@...` headers do not identify native categories.

## Native query regression checks

Install the SDK LangChain extra and pytest, then set `APERTUREDB_TEST_PORT` to an
isolated local server 1.0 and run `python -m pytest tests`. The checks execute
each notebook's real store constructor and metadata previews with deterministic
local embeddings; no LLM, Cohere or crawler requests are made.
