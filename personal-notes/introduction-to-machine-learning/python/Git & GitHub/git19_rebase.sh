# The "git rebase" command allows you to move or combine a series of commits onto a new base commit, effectively rewriting the commit history of a branch.

git rebase <branch_name>

# Allows you to interactively rebase your current branch onto <branch_name>, enabling you to edit, squash, reorder, or drop commits before integrating them into the target branch.

git rebase -i <branch_name>

# To initiates an interactive rebase session where you can modify the last specified number of commits from the current HEAD.

git rebase -i <HEAD~commits_numbers>

# "pick": Selects a commit to be included in the rebase without modification.
# "squash": Combines the selected commit with the one above it, allowing you to merge their changes into a single commit.
# "fixup": Similar to "squash", it combines the selected commit with the one above it, discarding its commit message in favor of the one above it.
# "edit": To edit the commit message
# "drop": Removes the selected commit from the rebase, effectively excluding it from the branch's history.
# "exec": Allows you to run shell commands during the rebase process, useful for performing custom actions or scripts.

