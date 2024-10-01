#!/bin/bash

# cd to the directory of the source code
cd "$(dirname "$0")/src"

# if build and dist directories exist, remove them
if [ -d "build" ]; then
  rm -r build
fi

# build and upload to pypi
python setup.py sdist bdist_wheel

# verify the package, upload if ok
twine check dist/* && twine upload dist/*

# remove build and dist directories
rm -r build dist