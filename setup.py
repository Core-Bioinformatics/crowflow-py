from setuptools import setup, find_packages
import os

__version__ = "1.0.0"

setup(
    name='crow',
    version=__version__,
    packages=find_packages(),
    description='Python package for evaluation of repeated stochastic clustering.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'seaborn',
        'plotnine',
        'ClustAssessPy'
    ],
    license='MIT',
    author="Rafael Kollyfas",
    author_email="rk720@cam.ac.uk",
    python_requires='>=3.7',
    keywords=['clustering', 'evaluation', 'stability', 'assessment', 'machine learning'],
    zip_safe=False,
)