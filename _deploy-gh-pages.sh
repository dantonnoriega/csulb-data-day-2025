#!/usr/bin/env bash

# Refresh / Create / Deploy gh-pages
# 1. checkout orphaned branch; remove README and .gitignore
git checkout --orphan gh-pages

# 2. generate slides
quarto render danton-csulb-data-day-2025.qmd --to revealjs -o slides.html --output-dir docs

# 3. add docs then commit
git add -f docs
git commit -m "slides render $(date '+%FT%T')" docs

# 4. delete upstream `gh-pages`
git push origin --delete gh-pages

# 5. push changes and come back to `main`
git subtree push --prefix docs origin gh-pages
git checkout main
git stash

# 6. remove gh-pages local branch
git branch -D gh-pages