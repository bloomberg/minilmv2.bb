"""Copyright 2022 Bloomberg Finance L.P.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup

setup(name='minilmv2',
      version='0.0.1',
      description='MiniLMV2 training and evaluation',
      url='https://github.com/bloomberg/minilmv2.bb', 
      author='Karthik Radhakrishnan',
      author_email='kradhakris10@bloomberg.net',
      python_requires='==3.7.*',
      license='Apache 2.0',
      packages=['minilmv2'],
      package_data={"": ["*.json"]},
      install_requires=['transformers==4.30.0', 'datasets==2.1.0', "torch==1.9.0", "awscli", "evaluate", "scipy", "scikit-learn"],
      zip_safe=False)
