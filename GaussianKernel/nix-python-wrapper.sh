#!/bin/sh
SCRIPT_DIR=$(dirname "$(realpath "$0")")
exec nix develop $SCRIPT_DIR --command python "$@"
