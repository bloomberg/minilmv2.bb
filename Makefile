# Copyright 2022 Bloomberg Finance L.P.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# VENV := .venv # If you are using a venv, replace this with path to your venv
# SOURCE_VENV := source $(VENV)/bin/activate;

package_dir := minilmv2
tests_dir := tests

PYEXEC := $(SOURCE_VENV) python
PYTEST_CMD:= $(PYEXEC) -m pytest --cov=$(package_dir) --cov-report=term-missing --cov-branch $(package_dir) $(tests_dir)
setup := $(PYEXEC) setup.py

.PHONY: all
all: fix-lint lint build test

.PHONY: black
black: 
	$(PYEXEC) -m black --check -q $(package_dir)
	$(PYEXEC) -m black --check -q $(tests_dir)

.PHONY: black-fix
black-fix: 
	$(PYEXEC) -m black -q $(package_dir)
	$(PYEXEC) -m black -q $(tests_dir)

.PHONY: isort-fix
isort-fix: 
	$(PYEXEC) -m isort -rc $(package_dir)
	$(PYEXEC) -m isort -rc $(tests_dir)

.PHONY: isort
isort: 
	$(PYEXEC) -m isort --check-only -rc $(package_dir)
	$(PYEXEC) -m isort --check-only -rc $(tests_dir)


.PHONY: pydocstyle
pydocstyle: 
	$(PYEXEC) -m pydocstyle --config setup.cfg $(package_dir)

.PHONY: flake8
flake8: 
	$(PYEXEC) -m flake8 --config=setup.cfg $(package_dir)

.PHONY: mypy
mypy: 
	$(PYEXEC) -m mypy --config-file setup.cfg $(package_dir)

.PHONY: lint
lint: isort flake8 mypy black pydocstyle

.PHONY: fix-lint
fix-lint: isort-fix black-fix


.PHONY: build
build:
	$(setup) build

.PHONY: test
test: 
	$(PYTEST_CMD)
