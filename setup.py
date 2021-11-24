from os import path
from setuptools import setup
from codecs import open # To use a consistent encoding


__package__ = 'bodegic'
__version__ = '0.2.0'
__licence__ = 'LGPL3'
__maintainer__ = 'Mehdi Golzadeh'
__email__ = 'golzadeh.mehdi@gmail.com'
__url__ = 'https://github.com/mehdigolzadeh/BoDeGiC'
__description__ = 'BoDeGiC - Bot detector an automated tool to identify bots in Git repositories by analysing commit messages'
__long_description__ = 'This tool accepts the name of a list of Git repositories and computes its output in three steps.\\\
The first step consists of extracting all commit information from the specified Git repositories using git log. This step results in a list of authors and their corresponding commits.\\\
The second step consists of computing the number of messages, empty messages, message patterns, and inequality between the number of messages within patterns.\\\
The third step simply applies the model we developed on these examples and outputs the prediction made by the model.'
__classifiers__=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
__requirement__ = [
        'python-dateutil >= 2.7.5',
        'pandas >= 0.23.4',
        'scikit-learn >= 0.22',
        'argparse >= 1.1',
        'GitPython >= 3.1.8',
        'tqdm >= 4.41.1',
        'urllib3 >= 1.25',
        'python-levenshtein >= 0.12.0',
        'numpy >= 1.17.4',
]

setup(
    name=__package__,

    version=__version__,

    description= __description__,
    long_description=__long_description__,

    url=__url__,

    maintainer=__maintainer__,
    maintainer_email=__email__,

    license=__licence__,

    classifiers=__classifiers__,

    keywords='git bot author commit message similarity',

    install_requires = __requirement__,

    include_package_data = True,
    packages = ['.'],

    entry_points={
        'console_scripts': [
            'bodegic=bodegic:cli',
        ]
    },

    py_modules=['bodegic'],
    zip_safe=True,
)
