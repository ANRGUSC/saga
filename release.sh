#!/usr/bin/env bash
set -euo pipefail

PYPROJECT="pyproject.toml"

# Get current version
current=$(grep -m1 '^version' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r major minor patch <<< "$current"

echo "Current version: $current"
echo ""
echo "Select bump type:"
echo "  1) patch  -> $major.$minor.$((patch + 1))"
echo "  2) minor  -> $major.$((minor + 1)).0"
echo "  3) major  -> $((major + 1)).0.0"
echo "  4) custom"
echo ""
read -rp "Choice [1]: " choice
choice="${choice:-1}"

case "$choice" in
    1) new_version="$major.$minor.$((patch + 1))" ;;
    2) new_version="$major.$((minor + 1)).0" ;;
    3) new_version="$((major + 1)).0.0" ;;
    4) read -rp "Enter version: " new_version ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "Bumping $current -> $new_version"
echo ""

# Update pyproject.toml
sed -i "s/^version = \"$current\"/version = \"$new_version\"/" "$PYPROJECT"
echo "Updated $PYPROJECT"

# Commit, tag, and push
git add "$PYPROJECT"
git commit -m "Bump version to $new_version"
git tag -a "v$new_version" -m "Release v$new_version"

read -rp "Push commit and tag to origin? [Y/n]: " push
push="${push:-Y}"
if [[ "$push" =~ ^[Yy]$ ]]; then
    git push origin HEAD
    git push origin "v$new_version"
    echo ""
    echo "Pushed. Creating GitHub release..."
    gh release create "v$new_version" --title "v$new_version" --generate-notes
    echo ""
    echo "Done! Release v$new_version created."
    echo "The publish workflow will now build and upload to PyPI."
else
    echo ""
    echo "Skipped push. When ready, run:"
    echo "  git push origin HEAD && git push origin v$new_version"
    echo "  gh release create v$new_version --title \"v$new_version\" --generate-notes"
fi
