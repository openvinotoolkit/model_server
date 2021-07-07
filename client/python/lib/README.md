# OpenVINO&trade; Model Server Client Library

Model server client library is a set of objects and methods designed to simplify user interaction with the instance of the model server. The library contains functions that hide API specific aspects so user doesn't have to know about creating protos, preparing requests, reading responses etc. and can focus on the application itself rather than dealing with API calls to OVMS.


## Building pip package

The library is distributed in form of pip package. To build and install the package first clone model server repository:

`git clone https://github.com/openvinotoolkit/model_server.git`

Then get to the client library directory:

`cd model_server/client/python/lib`

Make sure you have `setuptools` installed in your Python environment:

`pip install setuptools`

To build the wheel run:

`python setup.py bdist_wheel`

To install the wheel run:

`pip install dist/ovmsclient-0.1-py3-none-any.whl`

If all above steps are completed successfully you should be able to import and use the library in you Python environment.

## Generating documentation

The documentation is kept as docstrings in the code. The more readible, HTML form is generated from that docstrings with help of [sphinx](https://www.sphinx-doc.org/en/master/). 

To generate documentation first you need to install sphinx:

`pip install sphinx`

Then create `docs` catalog inside `lib` directory and in this new catalog run `sphinx-quickstart`:

```
mkdir docs
cd docs
sphinx-quickstart
```

Make sure that while passing the quickstart dialog you enable autodoc so sphinx can generate documentation from docstrings. 

When its's done open `conf.py` file and in the "Path setup" section replace commented code with:

```
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
```

In `index.rst` add `modules` under `toctree` section so the file looks somewhat like this:

```
.. OpenVINO Model Server Python client library documentation master file, created by
   sphinx-quickstart on Thu May 13 16:27:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenVINO Model Server Python client library's documentation!
=======================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

```

After it's done you can generate `.rst` files for the rest of the modules with:

`sphinx-apidoc -o . ..`

And then generate HTML files:

`make html`

The documentation pages should appear in `docs/_build/html`

### TODO
Generating documentation from docstring should be automated and exposed via make target like: `make docs`

