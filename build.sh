#!/bin/sh -xue
/bin/time podman run --rm \
  -v "$PWD":/code -w /code -e CARGO_HOME=/code/target/podmancargocache \
  rust:1-bullseye \
  ./containerbuild.sh

