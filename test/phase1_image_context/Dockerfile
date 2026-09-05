FROM ubuntu:24.04

WORKDIR /opt/thunder-phase1-smoke
COPY . .

RUN set -eux; \
    test "$(cat artifacts/message.txt)" = "thunder phase 1 image smoke test"; \
    grep -q '"contract": "phase1"' artifacts/build-info.json; \
    test ! -e ignored.txt; \
    test ! -e artifacts/ignored.tmp; \
    test ! -e ignored-directory/ignored.txt; \
    printf 'dockerfile build completed\n' > /opt/thunder-phase1-smoke/build-complete.txt

CMD ["/bin/bash", "-lc", "cat /opt/thunder-phase1-smoke/build-complete.txt && cat artifacts/message.txt"]
