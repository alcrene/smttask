"""
Utility functions for writing system tests.
"""
from builtins import zip
from builtins import str

import os.path
import re
from itertools import islice
import tempfile
import shutil
import sarge

DEBUG = False
temporary_dir = None
working_dir = None
env = {}


label_pattern = re.compile("Record label for this run: '(?P<label>\d{8}-\d{6})'")
label_pattern = re.compile("Record label for this run: '(?P<label>[\w\-_]+)'")

info_pattern = r"""Project name        : (?P<project_name>\w+)
Default executable  : (?P<executable>\w+) \(version: \d+.\d+.\d+\) at /[\w\/_.-]+/bin/python
Default repository  : MercurialRepository at \S+/sumatra_exercise \(upstream: \S+/ircr2013\)
Default main file   : (?P<main>\w+.\w+)
Default launch mode : serial
Data store \(output\) : /[\w\/]+/sumatra_exercise/Data
.          \(input\)  : /[\w\/]+/sumatra_exercise
Record store        : Django \(/[\w\/]+/sumatra_exercise/.smt/records\)
Code change policy  : (?P<code_change>\w+)
Append label to     : None
Label generator     : timestamp
Timestamp format    : %Y%m%d-%H%M%S
Plug-ins            : \[\]
Sumatra version     : 0.8dev
"""


def setup():
    """Create temporary directory for the Sumatra project."""
    global temporary_dir, working_dir, env
    temporary_dir = os.path.realpath(tempfile.mkdtemp())
    working_dir = os.path.join(temporary_dir, "sumatra_exercise")
    os.mkdir(working_dir)
    print(working_dir)
    env["labels"] = []


def teardown():
    """Delete all files."""
    if os.path.exists(temporary_dir):
        shutil.rmtree(temporary_dir)


def run(command):
    """Run a command in the Sumatra project directory and capture the output."""
    return sarge.run(command, cwd=working_dir, stdout=sarge.Capture(timeout=10, buffer_size=1))


def assert_file_exists(p, relative_path):
    """Assert that a file exists at the given path, relative to the working directory."""
    assert os.path.exists(os.path.join(working_dir, relative_path))


def pairs(iterable):
    """
    ABCDEF -> (A, B), (C, D), (E, F)
    """
    return zip(islice(iterable, 0, None, 2), islice(iterable, 1, None, 2))


def get_label(p):
    """Obtain the label generated by 'smt run'."""
    match = label_pattern.search(p.stdout.text)
    if match is not None:
        return match.groupdict()["label"]
    else:
        return None


def assert_in_output(p, texts):
    """Assert that the stdout from process 'p' contains all of the provided text."""
    if isinstance(texts, (str, type(re.compile("")))):
        texts = [texts]
    for text in texts:
        if isinstance(text, type(re.compile(""))):
            assert text.search(p.stdout.text), "regular expression '{0}' has no match in '{1}'".format(text, p.stdout.text)
        else:
            assert text in p.stdout.text, "'{0}' is not in '{1}'".format(text, p.stdout.text)


def assert_config(p, expected_config):
    """Assert that the Sumatra configuration (output from 'smt info') is as expected."""
    match = re.match(info_pattern, p.stdout.text)
    assert match, "Pattern: %s\nActual: %s" % (info_pattern, p.stdout.text)
    for key, value in expected_config.items():
        assert match.groupdict()[key] == value, "expected {0} = {1}, actually {2}".format(key, value, match.groupdict()[key])


def parse_records(text):
    records = []
    data = {}
    for line in text.split("\n"):
        if line[:5] == "-----":
            if data:
                records.append(data)
            data = {}
        else:
            first_column_width = line.find(':')
            first_column_content = line[:first_column_width].strip()
            value = line[first_column_width+1:].strip()
            if first_column_content:
                field_name = first_column_content.lower()
                data[field_name] = value
            else:
                data[field_name] += " " + value
    if data:
        records.append(data)
    patterns = {
        "repository": r'(?P<vcs>\w+)Repository at .*',
        "executable": r'(?P<executable_name>\w+) \(version: (?P<executable_version>[\w\.]+)\) at\s+(: )?(?P<executable_path>.*)'
    }
    for field_name, pattern in patterns.items():
        for record in records:
            match = re.search(pattern, record[field_name])
            for k, v in match.groupdict().items():
                record[k] = v
    return records


def assert_records(p, expected_records):
    """ """
    record_list = parse_records(p.stdout.text)
    records = dict((rec["label"], rec) for rec in record_list)
    for expected in expected_records:
        if expected["label"] not in records:
            raise KeyError("Expected record %s not found in %s" % (expected["label"], str(list(records.keys()))))
        matching_record = records[expected["label"]]
        for key in expected:
            assert expected[key] == matching_record[key]


def assert_return_code(p, value):
    assert p.returncode == value, "Return code {0}, expected {1}".format(p.returncode, value)


def assert_label_equal(p, expected_label):
    """ """
    assert get_label(p) == expected_label


def expected_short_list(env):
    """Generate the expected output from the 'smt list' command, given the list of captured labels."""
    return "\n".join(reversed(env["labels"]))


def substitute_labels(expected_records):
    """ """
    def wrapped(env):
        for record in expected_records:
            index = record["label"]
            record["label"] = env["labels"][index]
        return expected_records
    return wrapped


def build_command(template, env_var):
    """Return a function which will return a string."""

    def wrapped(env):
        args = env[env_var]
        if hasattr(args, "__len__") and not isinstance(args, str):
            s = template.format(*args)
        else:
            s = template.format(args)
        return s
    return wrapped


def edit_parameters(input, output, name, new_value):
    """ """
    global working_dir

    def wrapped():
        with open(os.path.join(working_dir, input), 'r') as fpin:
            with open(os.path.join(working_dir, output), 'w') as fpout:
                for line in fpin:
                    if name in line:
                        fpout.write("{0} = {1}\n".format(name, new_value))
                    else:
                        fpout.write(line)
    return wrapped


def run_test(command, *checks):
    """Execute a command in a sub-process then check that the output matches some criterion."""
    global env, DEBUG

    if callable(command):
        command = command(env)
    p = run(command)
    if DEBUG:
        print(p.stdout.text)
    if assert_return_code not in checks:
        assert_return_code(p, 0)
    for check, checkarg in pairs(checks):
        if callable(checkarg):
            checkarg = checkarg(env)
        check(p, checkarg)
    label = get_label(p)
    if label is not None:
        env["labels"].append(label)
        print("label is", label)
run_test.__test__ = False  # nose should not treat this as a test
