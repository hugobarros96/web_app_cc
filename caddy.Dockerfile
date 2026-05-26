# syntax=docker/dockerfile:1.7
FROM caddy:2-builder AS builder

# Cache mounts reuse the Go module cache and build cache across rebuilds.
# First build is still ~10 min on a small VM; subsequent rebuilds (e.g. adding
# another --with plugin or bumping a version) drop to seconds-to-minutes.
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg/mod \
    xcaddy build \
        --with github.com/mholt/caddy-ratelimit

FROM caddy:2

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
