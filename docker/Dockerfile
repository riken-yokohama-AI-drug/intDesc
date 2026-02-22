FROM python:3.9

COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/entrypoint.sh /usr/local/src/entrypoint.sh

RUN mkdir -p /usr/local/src/interaction_descriptor
COPY src/ /usr/local/src/interaction_descriptor/

RUN chmod +x /usr/local/src/entrypoint.sh && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENTRYPOINT ["/usr/local/src/entrypoint.sh"]

