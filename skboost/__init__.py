# -*- coding: utf-8 -*-
"""Release data for the skboost project."""

# -----------------------------------------------------------------------------
#  Copyright (c) 2015, Nedomkull Mathematical Modeling AB.
# -----------------------------------------------------------------------------

author = 'Henrik Blidh'
author_email = 'henrik.blidh@nedomkull.com'
maintainer = 'Henrik Blidh'
maintainer_email = 'henrik.blidh@nedomkull.com'
license = 'MIT'
description = "Boosting Algorithms compatible with scikit-learn"
url = 'https://bitbucket.org/nedomkull/milboost'
download_url = 'https://bitbucket.org/nedomkull/milboost'
platforms = ['Linux', 'Mac OSX', 'Windows XP/Vista/7/8']
keywords = ['Machine Learning', 'Boosting']
classifiers = [
                  'Development Status :: 3 - Alpha',
                  'Intended Audience :: Science/Research',
                  'Intended Audience :: Developers',
                  'License :: MIT',
                  'Topic :: Software Development',
                  'Topic :: Scientific/Engineering',
                  'Operating System :: Microsoft :: Windows',
                  'Operating System :: POSIX',
                  'Operating System :: Unix',
                  'Operating System :: MacOS',
                  'Programming Language :: Python',
                  'Programming Language :: Python :: 2',
                  'Programming Language :: Python :: 2.6',
                  'Programming Language :: Python :: 2.7',
                  'Programming Language :: Python :: 3',
                  'Programming Language :: Python :: 3.3',
                  'Programming Language :: Python :: 3.4',
              ],


# Version information.  An empty _version_extra corresponds to a full
# release.  'dev' as a _version_extra string means this is a development
# version.
_version_major = 0
_version_minor = 1
_version_patch = 3
# _version_extra = '.dev1'
_version_extra = 'a2'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor, _version_patch]

__version__ = '.'.join(map(str, _ver))
if _version_extra:
    __version__ += _version_extra

version = __version__  # backwards compatibility name
version_info = (_version_major, _version_minor, _version_patch, _version_extra)
