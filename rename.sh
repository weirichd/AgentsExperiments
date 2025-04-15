#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: ./rename.sh <new_package_name>"
    exit 1
fi

PACKAGE_NAME="$1"

echo "Renaming project to: $PACKAGE_NAME"

# Replace all instances of {{package_name}} except in rename.sh and Makefile
grep -rl '{{package_name}}' . --exclude=rename.sh --exclude=Makefile | xargs sed -i "s/{{package_name}}/$PACKAGE_NAME/g"

# Rename the package directory
mv src/{{package_name}} "src/$PACKAGE_NAME"

# Update .env IMAGE_BASE_NAME
sed -i "s/IMAGE_BASE_NAME=.*/IMAGE_BASE_NAME=$PACKAGE_NAME/" .env

echo "Rename complete!"
echo "Next: run 'make init'"
