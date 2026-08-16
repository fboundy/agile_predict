ARG PYTHON_VERSION=3.12-slim

FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OMP_NUM_THREADS 1
ENV OPENBLAS_NUM_THREADS 1
ENV MKL_NUM_THREADS 1
ENV NUMEXPR_NUM_THREADS 1

# install psycopg2 dependencies, plus nginx: gunicorn sync workers must sit
# behind a buffering proxy or a handful of slow readers occupy every worker.
# See docs/PRODUCTION_PERFORMANCE_FINDINGS.md.
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    nginx \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /code

WORKDIR /code

COPY requirements.txt /tmp/requirements.txt
RUN set -ex && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/
COPY . /code

RUN SECRET_KEY=collectstatic-build-placeholder python manage.py collectstatic --noinput

# Replace Debian's nginx.conf outright — ours defines the whole http block, so
# the packaged default site in sites-enabled/ is never included.
# The CR strip is belt-and-braces: this repo is edited over a Samba share with
# core.autocrlf=true, and a CRLF shebang would make the entrypoint unrunnable
# in a way that only shows up at machine boot.
RUN cp /code/deploy/nginx.conf /etc/nginx/nginx.conf && \
    sed -i 's/\r$//' /code/deploy/web-entrypoint.sh /etc/nginx/nginx.conf && \
    chmod +x /code/deploy/web-entrypoint.sh && \
    nginx -t -c /etc/nginx/nginx.conf

EXPOSE 8000

# gunicorn's flags now live in the entrypoint rather than being duplicated here
# and in fly.toml, which had to be hand-synced since Aug 2.
CMD ["/code/deploy/web-entrypoint.sh"]
