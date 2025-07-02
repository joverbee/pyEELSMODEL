.. _installing:


Installing
==========

Installing via Conda
^^^^^^^^^^^^^^^^^^^^^^
To do

Installing via pip
^^^^^^^^^^^^^^^^^^
The easiest way to install pyEELSMODEL is through the release hosted on PyPI:

.. code-block:: bash

    $ pip install pyEELSMODEL

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
pyEELSMODEL can be installed by cloning the repository to your
computer via:

.. code-block:: bash

    $ git clone https://github.com/joverbee/pyEELSMODEL.git

The next step is to navigate to the pyEELMODEL directory and type
following into the command line:

.. code-block:: bash

    $ pip3 install .

If you want to create an editable install one needs to do following:

.. code-block:: bash

    $ pip3 install -e .

Setting up GOS tables
^^^^^^^^^^^^^^^^^^^^^
**When they are first needed, the generalized oscillator strengths (GOS) tables will be automatically imported.**
The GOS tables are necessary to perform EEL quantification since they are used
to calculate the atomic cross sections. Two different GOS tables can be used for quantification:

1. The GOS tables calculated by Zhang Z. *et al.* which can be found at doi:`10.5281/zenodo.7729585 <https://doi.org/10.5281/zenodo.7729585>`_.
2. The GOS tables calculated by Segger L. *et al.* which can be found at doi:`10.5281/zenodo.7645765 <https://doi.org/10.5281/zenodo.7645765>`_.


The GOS tables can also be manually imported. 
To know where your pyEELSMODEL package is installed following command can be run
in a python console:

.. code-block:: python

    import pyEELSMODEL
    print(pyEELSMODEL.__path__)

This information is necessary for the proper use of the GOS tables.

GOS tables from Zhang Z.
------------------------
Following steps explain on how the properly setup the GOS array of
Zhang Z.

1. Download the **Dirac_GOS_database.zip** file
2. Unzip the file
3. Copy the **.hdf5 files** in the folder to the **.pyEELSMODEL\\database\\Zhang** folder which is found in the pyEELSMODEL folder

**The GOS tables from Zhang Z. are used as default in the quantification workflows.**

GOS tables from Segger L.
------------------------------------
Following steps explain on how the properly setup the GOS array of
Segger L:

1. Download the **Segger_Guzzinati_Kohl_1.5.0.gosh** (depends on version) file
2. Copy the **.gosh file** to **.pyEELMODEL\\database\\Segger_Guzzinati_Kohl** folder which is found in the pyEELSMODEL folder

**Both cross sections are used in the notebook examples hence both need to be incorporated**
