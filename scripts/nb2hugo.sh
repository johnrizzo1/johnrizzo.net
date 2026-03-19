#!/usr/bin/env bash
#
# Convert a Jupyter notebook to a Hugo-compatible markdown post (page bundle).
#
# Usage: ./scripts/nb2hugo.sh path/to/notebook.ipynb [title] [tags]
#
# Examples:
#   ./scripts/nb2hugo.sh notebooks/analysis.ipynb
#   ./scripts/nb2hugo.sh notebooks/analysis.ipynb "My Analysis" "python,data"
#
# Output: content/posts/<notebook-name>/index.md  (plus any images)

set -euo pipefail

NOTEBOOK="$1"
TITLE="${2:-$(basename "${NOTEBOOK}" .ipynb | tr '_-' '  ')}"
TAGS="${3:-jupyter}"
DATE=$(date +%Y-%m-%d)

if [ ! -f "$NOTEBOOK" ]; then
  echo "Error: Notebook not found: $NOTEBOOK"
  exit 1
fi

if ! command -v jupyter &> /dev/null; then
  echo "Error: jupyter is not installed. Install with: pip install jupyter"
  exit 1
fi

# Derive post directory name from notebook filename
SLUG=$(basename "${NOTEBOOK}" .ipynb | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
POST_DIR="content/posts/${SLUG}"

mkdir -p "${POST_DIR}"

# Convert notebook to markdown, outputting into the post directory
jupyter nbconvert --to markdown --output-dir="${POST_DIR}" --output="index" "${NOTEBOOK}"

# Build the TOML frontmatter
IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
TAG_LIST=""
for tag in "${TAG_ARRAY[@]}"; do
  tag=$(echo "$tag" | xargs)  # trim whitespace
  TAG_LIST="${TAG_LIST}  \"${tag}\",
"
done

FRONTMATTER="+++
draft = false
authors = [\"John Rizzo\"]
title = \"${TITLE}\"
date = \"${DATE}\"
tags = [
${TAG_LIST}]
categories = [
  \"jupyter\"
]
series = []
+++"

# Prepend frontmatter to the converted markdown
TMPFILE=$(mktemp)
echo "${FRONTMATTER}" > "${TMPFILE}"
echo "" >> "${TMPFILE}"
cat "${POST_DIR}/index.md" >> "${TMPFILE}"
mv "${TMPFILE}" "${POST_DIR}/index.md"

echo "Created Hugo post: ${POST_DIR}/index.md"
echo "Review the post and adjust frontmatter as needed."
