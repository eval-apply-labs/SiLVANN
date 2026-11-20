---
name: files
description: find files and folders by name under a folder, and the lines of a file that hold a pattern
---
A skill the release ships, and an example of one: a command in Python, which asks for the folder it reads as `ls` does —
unasked in the workspace and the folders you allowed, asked elsewhere.

## (find "folder" "pattern")
touches: none
run: files.py:find
The names under the folder, and under its folders, that match the pattern — `"*.txt"`, `"notes*"`, `"*"` — as a list
of paths relative to the folder; a folder's ends in `/`. Folders whose names start with a dot are not looked into. At
most 500, then a last element that says how many more: a narrower folder or pattern finds them.

## (grep "pattern" "file" [from])
touches: none
run: files.py:grep
The lines of the file that hold the pattern, each with its number before it — a regular expression, capitals and small
letters alike. At most 50, then a last element that says how to go on, `(grep "pattern" "file" from)` answering from
line `from`. `(read "file" number count)` reads the lines around one found. It is how a long file is looked into
without reading it whole, a document sent with a message and kept in `attached/`, a log, a book.
