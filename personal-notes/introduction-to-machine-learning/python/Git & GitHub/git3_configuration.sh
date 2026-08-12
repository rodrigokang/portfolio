# Set credentials

git config --global user.name "rodrigokang" # usuario
git config --global user.email "rodrigokang88@gmail.com" # correo

# User and email

git config --global -e

# Shortcuts

# To create a global Git alias, allowing you to use the shorthand "git s" to display a concise status summary with branch information.

git config --global alias.s "status -s -b"

# To create a global Git alias, enabling you to use the shortcut "git l" to display a compact log with one-line commit messages, decorations, all branches, and a graphical representation of commit history.

git config --global alias.l "log --oneline --decorate --all --graph"