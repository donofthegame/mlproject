from setuptools import find_packages, setup
from typing import List

hypen = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    will return the list of requirements
    '''
    req=[]
    with open(file_path) as file_obj:
        req=file_obj.readlines()
        req=[r.replace('\n','') for r in req]
        if hypen in req:
            req.remove(hypen)
    return req

setup(
    name='mlproject',
    version='0.0.1',
    author='Mynu',
    author_email='vragarsha@gmail.com',
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'seaborn'] # this works when the packages are few
    install_requires=get_requirements('requirements.txt')
)