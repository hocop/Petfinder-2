import codecs
import os.path
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1].strip()
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='pf2',
    version=get_version('pf2/__init__.py'),
    description='Deep Unsupervised Optical Flow in Pytorch',
    long_description='Deep Unsupervised Optical Flow in Pytorch',
    long_description_content_type='text/markdown',
    author='Ruslan Baynazarov',
    author_email='ruslan.baynazarov@fastsense.tech',
    url='',
    packages=[
        'pf2',
        'pf2.utils',
        'pf2.networks',
    ],
    license="MIT",
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
        'albumentations',
        'pytorch-lightning',
        'torchmetrics',
        'einops',
        'wandb',
    ]
)
