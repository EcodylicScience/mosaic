"""Name the settings keys that configure `remote`, and the characters that
their values may use."""

# The keys that `remote` requires, in the environment or under `env` in the
# settings file.
REQUIRED_KEYS = ("REMOTE_TASK_HOST", "REMOTE_TASK_DIR")
CONFIGURATION_KEYS = (*REQUIRED_KEYS, "REMOTE_TASK_TIMEOUT")

# Regex class bodies. Values that use only these characters can be
# interpolated into a remote command line unquoted.
HOST_CHARACTERS = "A-Za-z0-9._@-"
DIRECTORY_CHARACTERS = "A-Za-z0-9._@/-"
