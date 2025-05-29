
from setuptools import setup, find_packages
from typing import List
# This file is used to create a package for the project
# It will be used to install the package using pip
# The requirements.txt file will be used to install the dependencies
## create a function that will return a list of packages 
def get_requirements(file_path:str) -> List:
    """
    This function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements 

setup(
    name='mlproject',
    version='0.0.1',
    author='YASSIR',
    author_email='absisi2009@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A machine learning project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    
)