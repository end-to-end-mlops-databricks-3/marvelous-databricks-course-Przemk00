#!/bin/bash

echo "INIT SCRIPT: Starting Git authentication setup..."
echo "INIT SCRIPT: Current user: $(whoami)"
echo "INIT SCRIPT: HOME directory: $HOME"
echo "INIT SCRIPT: PATH: $PATH"
echo "INIT SCRIPT: Checking for git command..."
if command -v git &> /dev/null
then
    echo "INIT SCRIPT: git command found at $(command -v git)"
    git --version
else
    echo "INIT SCRIPT ERROR: git command NOT FOUND!"
fi

# Ensure the GIT_TOKEN environment variable is set in the cluster
if [ -z "$GIT_TOKEN" ]; then
  echo "INIT SCRIPT ERROR: GIT_TOKEN environment variable is NOT SET on the cluster."
  echo "INIT SCRIPT: Git authentication for github.com via .netrc will fail."
  exit 0 # Exit gracefully, but problem will persist
fi

echo "INIT SCRIPT: GIT_TOKEN is set. First 5 chars: ${GIT_TOKEN:0:5}..." # Avoid logging full token

# Define the path for .netrc.
NETRC_FILE_TARGET="/root/.netrc"
echo "INIT SCRIPT: Target .netrc path: $NETRC_FILE_TARGET"

# Attempt to write to .netrc
# Create the directory if it doesn't exist (though /root should exist)
mkdir -p "$(dirname "$NETRC_FILE_TARGET")"
echo "machine github.com login $GIT_TOKEN password x-oauth-basic" > "$NETRC_FILE_TARGET"
WRITE_EXIT_CODE=$?

if [ $WRITE_EXIT_CODE -ne 0 ]; then
    echo "INIT SCRIPT ERROR: Failed to write to $NETRC_FILE_TARGET. Exit code from echo redirect: $WRITE_EXIT_CODE"
    ls -ld "$(dirname "$NETRC_FILE_TARGET")" # Check directory permissions
    exit 1 # Indicate an error occurred
fi

chmod 600 "$NETRC_FILE_TARGET"
CHMOD_EXIT_CODE=$?
if [ $CHMOD_EXIT_CODE -ne 0 ]; then
  echo "INIT SCRIPT ERROR: chmod 600 on $NETRC_FILE_TARGET failed. Exit code: $CHMOD_EXIT_CODE"
fi

if [ ! -f "$NETRC_FILE_TARGET" ]; then
  echo "INIT SCRIPT ERROR: $NETRC_FILE_TARGET was NOT created successfully."
  ls -la /root/ # List contents of /root for debugging
  exit 1 # Indicate an error
fi

echo "INIT SCRIPT: Successfully created or updated $NETRC_FILE_TARGET."
echo "INIT SCRIPT: Verifying content of $NETRC_FILE_TARGET (login part only):"
grep "login" "$NETRC_FILE_TARGET" # This will show the line with the token, use with caution

echo "INIT SCRIPT: Git authentication setup finished."
