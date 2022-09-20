rm -r build/ dist/ *.egg-info

python setup.py bdist_wheel
twine upload --repository pypi dist/*
