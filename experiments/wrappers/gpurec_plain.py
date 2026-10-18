"""The gpurec command line, unchanged: every timed run of the paper starts gpurec this way, from the Python environment
that runs the experiment drivers. usage: gpurec_plain.py <gpurec arguments...>"""
import sys

sys.argv = ["gpurec"] + sys.argv[1:]
from gpurec.cli.main import entry

entry()
