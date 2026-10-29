"""Define the starting prompts that launch-gate fills in for the review-prose
peers.

Each `__NAME__` placeholder is replaced by a value of the run.
"""

EDITOR_PROMPT = """\
Target:

__TARGET__

Repository: __REPO__
Ruleset: __RULESET__
House style: __HOUSE_STYLE__
State file: __STATE__
Reviewer session: __REVIEWER__
Pathspec: __PATHS__
Range: __RANGE__

Diff command: git -C "__REPO__" diff -M --find-renames=40% __RANGE__

Wait for the reviewer's first findings.
"""

REVIEWER_PROMPT = """\
Target:

__TARGET__

Repository: __REPO__
Ruleset: __RULESET__
House style: __HOUSE_STYLE__
State file: __STATE__
Editor session: __EDITOR__
Caller session: __CALLER__
Pathspec: __PATHS__
Range: __RANGE__
Token: __TOKEN__

Diff command: git -C "__REPO__" diff -M --find-renames=40% __RANGE__

Start now. Read __RULESET__, run the diff command, and send stage one's
findings to __EDITOR__.
"""
