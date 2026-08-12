# Add a remote repository named 'origin' with the provided URL address.
git remote add origin <url_address>

# Rename the current branch to 'main'.
git branch -M main

# Push local changes to the remote repository.
git push

# Push local changes to the remote repository's main branch, setting it as the default upstream branch.
git push -u origin main

# Push tags to the remote repository.
git push --tags

# List all remote repositories along with their URLs.
git remote -v

# Fetch changes from the remote repository without merging them into the local branch.
git fetch

# Fetch changes from the remote repository and merge them into the current branch.
git pull

# Clone a repository from the provided URL address to the local machine.
git clone <url_address>

# Push local changes to the remote repository, setting up the upstream branch if it doesn't exist.
git push --set-upstream origin <branch_name>