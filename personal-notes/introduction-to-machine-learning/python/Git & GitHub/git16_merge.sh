# In Git, "merge" is a command used to combine the changes from one branch (either an auxiliary branch or a secondary branch) into another, such as the main branch. This merges the changes made in the selected branch into the target branch, thus integrating the work done in both branches.

# A "fast-forward" merge occurs when the target branch of a merge is directly ahead of the current branch, allowing Git to simply move the current branch pointer forward to match the target branch's latest commit.

# Integrate changes from the specified branch into the current branch.

git merge <branch_name>

# To delete the specified branch  if it has been fully merged into the current branch, ensuring a clean and streamlined branch history. Generally considered a good practice to delete branches that are no longer needed, especially after they have been merged into the main development branch. This helps to keep the repository tidy and avoids cluttering the branch list with branches that have served their purpose.

git branch -d <branch_name>