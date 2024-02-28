.. _pyEELSMODEL:

What is pyEELSMODEL
===================

pyEELSMODEL is an EELS analysis package derived from the former `eelsmodel <https://github.com/joverbee/eelsmodel>`_ which
was written in c++. This package uses the model-based approach to quantify EEL spectra.

pyEELSMODEL is a package which is written mainly in python language to make it
more readable and understandable this comprimises in terms of speed but improves
understanding of how the data was treated (algorithms are no black box).

This package uses a minimal amount of dependencies (matplotlib, numpy, h5py and sci-kit)
which should make the package more stable over time without having conflicts when trying
to install other packages.


The package of pyEELSMODEL contains a back-bone structure where spectrums, components
and fitters are defined. These classes are then used to perform operations on
the experimental data, the operations are defined as new classes.

EEE


