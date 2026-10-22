# Linux deployment recipe only. Supply a reviewed Python 3.14 image pinned by digest.
ARG MEMORY_BASE_IMAGE
FROM ${MEMORY_BASE_IMAGE}
WORKDIR /opt/hermes
COPY . /opt/hermes
RUN python -m pm.build_env --source /opt/hermes --out /opt/hermes-env
ENV PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
ENTRYPOINT ["/opt/hermes-env/bin/python", "-m", "local_extensions.e4b_memory.restricted"]
