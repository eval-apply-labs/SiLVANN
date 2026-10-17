---
name: files
description: find files and folders by name under a folder
---
A skill the release ships, and an example of one: a command in Python, which asks for the folder it reads as `ls` does —
unasked in the workspace and the folders you allowed, asked elsewhere.

## (find "folder" "pattern")
touches: none
run: files.py:find
The names under the folder, and under its folders, that match the pattern — `"*.txt"`, `"notes*"`, `"*"` — as a list
of paths relative to the folder; a folder's ends in `/`. Folders whose names start with a dot are not looked into. At
most 500, then a last element that says how many more: a narrower folder or pattern finds them.
