# To create a lightweight tag at the current commit in Git, without adding any extra information like a message or timestamp.

git tag <tag_name>

# Delete the specified tag from the Git repository.

git tag -d <tag_name>

# Create an annotated tag with the specified <version_name>, allowing you to add additional information such as a message and timestamp.

git tag -a <tag_name>

# Create an annotated tag named <tag_name> at the specified commit <hash_serie>, with the provided <description> as the tag message.

git tag -a <tag_name> <hash_serie> -m <description>

# Display the detailed information, including the commit associated with the specified tag, any changes made, and the commit message.

git show <tag_name>