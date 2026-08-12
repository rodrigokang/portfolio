# To temporarily shelves (or stashes) changes made to the working directory and index, allowing you to work on another task or branch without committing the changes.

git stash # Note: WIP means "Work in Progress"

# Display a list of all stashed changes, showing their respective stash IDs along with a description if provided.

git stash list

# Apply the most recently stashed changes to the working directory and removes them from the stash list.

git stash pop

# Remove the most recently stashed changes from the stash list, permanently discarding them.

git stash drop

# Display the changes that are currently stashed, showing the file names and the lines that have been modified.

git stash show

# Remove all stashed changes from the stash list, effectively clearing the stash and permanently discarding all stashed changes.

git stash clear

# Apply the most recently stashed changes to the working directory without removing them from the stash list, allowing you to apply the changes multiple times if needed.

git stash apply

# Apply the changes from the specified stash to the working directory without removing them from the stash list, identified by the <stash_id>

git stash apply <stash_id>

# Display a list of stashed changes along with a summary of the changes introduced by each stash.

git stash list --stat

# Displays the changes in the stashed commit along with the diff details.

git stash show -p

# To create a new branch from the specified stash and applies the changes stored in that stash onto the new branch, removing them from the stash.

git stash branch <brach_name> <stash_id>