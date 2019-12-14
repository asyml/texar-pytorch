import sys
import setuptools

long_description = """
Texar-PyTorch is an open-source toolkit based on PyTorch,
aiming to support a broad set of machine learning especially text generation
tasks, such as machine translation, dialog, summarization, content manipulation,
language modeling, and so on.

Texar is designed for both researchers and practitioners for fast prototyping
and experimentation. Checkout https://github.com/asyml/texar for the TensorFlow
version which has the same functionalities and (mostly) the same interfaces.
"""

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Texar-PyTorch.')

setuptools.setup(
    name="texar-pytorch",
    version="0.1.1-unreleased",
    url="https://github.com/asyml/texar-pytorch",

    description="Toolkit for Machine Learning and Text Generation",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=[
        f"texar.{name}"
        for name in setuptools.find_packages(where='texar/')
    ],
    platforms='any',

    install_requires=[
        'regex>=2018.01.10',
        'numpy',
        'requests',
        'funcsigs',
        'sentencepiece>=0.1.8',
        'mypy_extensions',
        'packaging>=19.0'
    ],
    extras_require={
        'torch': ['torch>=1.0.0'],
        'examples': [],
        'extras': ['Pillow>=3.0', 'tensorboardX>=1.8'],
    },
    package_data={
        "texar.torch": [
            "../../bin/utils/multi-bleu.perl",
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
