# How to Contribute
I welcome collaboration on this project. I need the most help with keeping up with documentation, writing more unittests, and adding other ASI missions. Please contact me or start a Pull Request with your suggestions. 

To install the developer dependencies, clone this repo, `cd aurora-asi-lib` and then run `python3 -m pip install -r requirements.txt -e .`

# PyPI Release Checklist
- [ ] Commit your latest changes:
- [ ] Style with black:
```
cd aurora-asi-lib
python3 -m black -l 100 -S asilib/
```
- [ ] Update version number (can also be minor or majorl; this will generate a new tag v`MAJOR`.`MINOR`.`PATCH`):
```
bumpversion patch
```
- [ ] Run unit tests and verify that all tests pass:
```
cd aurora-asi-lib
python3 -m unittest discover -v
```
- [ ] Push: `git push`
- [ ] Push tags: `git push --tags`
- [ ] Create a [new release](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) on GitHub with the newest tag. This triggers the upload to PyPI.
- [ ] Check the aurora-asi-lib PyPI page to make sure that the README and the version number are correct. 
<!-- TODO: Add instructions to upload to test PyPI -->
- [ ] Lastly, a sanity check that the PyPI version works:
```
python -m venv env
source env/bin/activate
python3 -m pip install aurora-asi-lib
<import asilib in python3 interpreter>
deactivate
rm -r env/
```

## Test
To run the asilib unit tests, change directory into `asi-lib` and run ```python3 -m unittest discover -v```. These tests take a few minutes to run because it must download REGO and THEMIS image files. 

These tests are continously intergrated when the `main` branch is updated by a GitHub runner (virtual machine) using `pytest` and the configuration is saved in `.github/workflows/ci.yml`.

## Style and with black
I adoped the [black](https://pypi.org/project/black/) style with two modifications: line length is set to 100 characters, and I suppress the double-quote string setting. To run black from the `aurora-asi-lib` directory, run 

```python3 -m black -l 100 -S asilib/```.

## Change version
Read this entire section before running `bumpversion`. To change the version you will need to use `bumpversion` to bump the version by a major X.0.0, minor, 0.X.0, or patch 0.0.X (where X is incremented). Call ```bumpversion [major|minor|patch]``` in the command line to increment the version number. When you run this command, you should push the automatically created tag (`git push origin tag vX.Y.Z`,) commit to GitHub, and create a new release on GitHub to trigger an upload to 

__CAUTION:__ `config.py` should not be a part of the distribtuion on PyPI. When GitHub packages `aurora-asi-lib`, it __does not__ create a `config.py` (however it does when running the CI) so it is not an issue. If somehow `config.py` ends up on the GitHub repo, it will be packaged. However, if you package `aurora-asi-lib` on your local machine (via ```python setup.py sdist bdist_wheel```), `config.py` will be included in the package (unwanted behavior if you want to share it). If you think you know better by excluding `config.py` via a `MANIFEST.in`, be warned that there is a [bug](https://github.com/pypa/setuptools/issues/511): the source distribution will not have `config.py` __but__ the wheel will.