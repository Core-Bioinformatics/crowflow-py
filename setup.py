from setuptools import setup, find_packages
import os

__version__ = "1.0.0"

setup(
    name='crowflow',
    version=__version__,
    packages=find_packages(),
    description='Python package for evaluating clustering stability through the use of repeated stochastic clustering and element-centric evaluation metrics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'seaborn',
        'plotnine',
        'ClustAssessPy'
    ],
    url='https://github.com/Core-Bioinformatics/crowflow-py',
    license='MIT',
    author='Rafael Kollyfas',
    author_email='rk720@cam.ac.uk',
    python_requires='>=3.7',
    keywords=['clustering', 'evaluation', 'stability', 'assessment', 'machine learning'],
    zip_safe=False,
)