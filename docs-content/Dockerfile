FROM alpine:3
RUN apk add --no-cache bash rsync
COPY notebooks/ /content/notebooks/
COPY docs-content/publish.sh /publish.sh
ENV DOCS_PROVIDER=aperturedb-applications
ENTRYPOINT ["bash", "/publish.sh"]
