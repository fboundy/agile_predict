#!/bin/bash
# web-entrypoint.sh — run gunicorn behind an nginx buffering proxy.
#
# Two processes in one container, so the container has to supervise them. The
# requirement is that if *either* dies the machine goes down and Fly replaces
# it, rather than limping along with half the stack: a live nginx in front of a
# dead gunicorn serves 502s indefinitely, which the health check would catch
# eventually, but a dead nginx in front of a live gunicorn is simply invisible.
#
# `wait -n` returns on the first child to exit, which gives exactly that with no
# process manager. It is bash 4.3+ and not POSIX — this must not be /bin/sh,
# since Debian's dash does not implement it.

set -euo pipefail

SOCKET=/run/gunicorn.sock
rm -f "$SOCKET"

gunicorn \
    --bind "unix:$SOCKET" \
    --umask 0 \
    --workers "${GUNICORN_WORKERS:-4}" \
    --timeout 60 \
    --graceful-timeout 30 \
    --max-requests 800 \
    --max-requests-jitter 200 \
    config.wsgi &
gunicorn_pid=$!

# Avoid serving 502s during the boot window before gunicorn has bound.
for _ in $(seq 1 100); do
    [ -S "$SOCKET" ] && break
    sleep 0.1
done

nginx -c /etc/nginx/nginx.conf -g 'daemon off;' &
nginx_pid=$!

shutdown() {
    kill -TERM "$gunicorn_pid" "$nginx_pid" 2>/dev/null || true
}
trap shutdown TERM INT

# `set +e` so a non-zero child status reaches the cleanup below rather than
# tripping `set -e` and orphaning the surviving process.
set +e
wait -n
status=$?
set -e

shutdown
exit "$status"
