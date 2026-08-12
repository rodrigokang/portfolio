# Display the current state of the repository, including pending changes, untracked files, and the current branch.

git status

# To provide a short and concise summary of the Git repository's current status, displaying information about changes in the working directory and staging area. The output includes two-letter codes indicating the status of each file (e.g., M for modified, A for added, D for deleted) along with the file paths.

git status -s

# To not only provide a short and concise summary of the Git repository's current status, indicating changes in the working directory and staging area, but it also includes information about the current branch. The output includes a two-letter code indicating the status of each file, along with the file paths, and also indicates the current branch with additional information, such as <## branch-name>.

git status -s -b
