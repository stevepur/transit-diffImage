import codecs
import os.path

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "rt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="transit-diffimage",
    version=get_version("transitDiffImage/__init__.py"),
    author="Steve Bryson",
    author_email="steve.bryson@nasa.gov",
    url="https://github.com/stevepur/transit-diffImage",
    license="GPLv3",
    packages=["transitDiffImage"],
    package_dir={"transitDiffImage": "transitDiffImage"},
    package_data={"transitDiffImage": [
        "data/de432s.bsp",
        "data/TESS_merge_ephem.bsp",
        "data/tess2018338154046-41240_naif0012.tls",
    ]},
    install_requires=requirements,
)
