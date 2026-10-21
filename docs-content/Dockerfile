FROM alpine:3
RUN apk add --no-cache bash rsync
COPY notebooks/ /content/notebooks/
# CI supplies the release Git tag used for source/Colab links.
ARG SOURCE_REVISION=
RUN printf '%s\n' "$SOURCE_REVISION" > /content/.docs-source-revision
COPY docs-content/publish.sh /publish.sh
ENV DOCS_PROVIDER=aperturedb-applications
ENTRYPOINT ["bash", "/publish.sh"]
