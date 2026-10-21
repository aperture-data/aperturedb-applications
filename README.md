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
