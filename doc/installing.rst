.. _installing:


Installing
==========

Installing via Conda
^^^^^^^^^^^^^^^^^^^^^^
In progress

Installing via pip
^^^^^^^^^^^^^^^^^^
In progress

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
At this moment, pyEELSMODEL is installed by cloning the repository to your
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

**In the future, the package will be published on PyPI simplifying the
installation procedure.**

Setting up GOS tables
^^^^^^^^^^^^^^^^^^^^^
After installation, the generalized oscillator strengths (GOS) tables should be imported.
The GOS tables are necessary to perform EEL quantification since they are used
to calculate the atomic cross sections. Two different GOS tables can be used for quantification:

1. The GOS tables calculated by Zhang Z. *et al.* which can be found at doi:`10.5281/zenodo.7729585 <https://doi.org/10.5281/zenodo.7729585>`_.
2. The GOS tables calculated by Segger L. *et al.* which can be found at doi:`10.5281/zenodo.7645765 <https://doi.org/10.5281/zenodo.7645765>`_.


GOS tables from Zhang Z.
------------------------
Following steps explain on how the properly setup the GOS array of
Zhang Z.

1. Download the **Dirac_GOS_database.zip** file
2. Unzip the file
3. Copy the **.hdf5 files** in the folder to the **.\\database\\Zhang** folder which is found in the pyEELSMODEL folder

**The GOS tables from Zhang Z. are used as default in the quantification workflows.**

GOS tables from Segger L.
------------------------------------
Following steps explain on how the properly setup the GOS array of
Segger L:

1. Download the **Segger_Guzzinati_Kohl_1.5.0.gosh** (depends on version) file
2. Copy the **.gosh file** to **.\\database\\Segger_Guzzinati_Kohl** folder which is found in the pyEELSMODEL folder

**Both cross sections are used in the notebook examples hence both need to be incorporated**