# Thesis

## Dev requirements

Development is best done on Ubuntu 24.04.
It's best to use a new separate VM to avoid issues.

On Windows you can use WSL:

- <https://apps.microsoft.com/detail/ubuntu-22-04-2-lts/9PN20MSR04DW>

- Make sure you have systemd enabled, this should be enabled automatically, see: <https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl/#ensuring-you-are-on-the-right-wsl-version>

On MacOS or Linux you can use lima:

- <https://github.com/lima-vm/lima/blob/master/examples/ubuntu-lts.yaml>

## Dev setup

Update and install git

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git
```

Create ssh key for git

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

You can clone the repository now.
Ensure you are in the root of the project directory for these commands

```bash
# Set up dev environment
# NOTE: This will prompt for admin password at the end
./bin/setup.sh
# NOTE: Restart your shell after this

# Allow direnv to automatically create + activate venv
# and set environment variables when in project directory
direnv allow

# Install dependencies
make dep_sync

# Periodically update to latest packages to be in sync
sudo apt full-upgrade -y

# Show available dev commands
make

# Start jupyter notebook
make dev

# Use the following command to:
# Sync deps and start jupyter notebook
make dev_full

# To see outdated dependencies and update to latest versions
# allowed in pyproject.toml
make dep_outdated
make dep_upgrade
```
