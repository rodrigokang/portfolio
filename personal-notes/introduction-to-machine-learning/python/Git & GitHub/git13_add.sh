# Remove files from the staging area and also deletes them from the working directory.

git rm <file_name>

# To move or rename files or directories within a Git repository while simultaneously staging the changes.

git mv <file_name> (<directory_name>)

# Stage all modified and deleted files, but not untracked files, for the next commit.

git add -u <file_name>

git commit -m <"Removing scripts">