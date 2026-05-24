"""Test fixtures for scheduler tests.

The pure-Python agent nodes don't call OpenAI, but `SchedulingAgent.__init__`
constructs a `ChatOpenAI` client that validates the API key field exists.
Provide a fake key if the environment doesn't already have one so the tests
can run without secrets.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-pure-python-tests")
