#!/bin/bash

# cd to the directory of this script
cd "$(dirname "$0")"

# build and upload to pypi
python setup.py sdist bdist_wheel

# verify the package, upload if ok
twine check dist/* && twine upload dist/*