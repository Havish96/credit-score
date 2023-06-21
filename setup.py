from setuptools import find_packages
from setuptools import setup

with open('prod_requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='credit_score',
      description="Credit Score Classification",
      author="Le Wagon Batch #1247",
      author_email="soowamhc@gmail.com",
      url="https://github.com/Havish96/credit-score",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
