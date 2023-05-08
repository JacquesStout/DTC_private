from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.6.0'
DESCRIPTION = 'Diffusion-images To Connectome processing toolbox'
LONG_DESCRIPTION = 'Set of tools for processing of diffusion images into connectome form using dipy methods and others'

# Setting up
setup(
    name="DTC",
    version=VERSION,
    author="Jacques Andrew Stout",
    author_email="jacques.stout@duke.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=['pyautodep'],
    zip_safe=False,
    install_requires=['opencv-python', 'pyautogui'],
    keywords=['python', 'package', 'autoinstall', 'pypi package'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
        'console_scripts': [
            'pyautodep=pyautodep.__main__:main'
        ]
    }
)