.. _intro:

What is pyEELSMODEL
===================

pyEELSMODEL is an EELS analysis package derived from the former `eelsmodel <https://github.com/joverbee/eelsmodel>`_ which
was written in c++. This package uses the model-based approach to quantify EEL spectra. Additionally, more traditional
methods such as power-law subtraction and deconvolution are implemented to have an easy way of comparing the different
methodologies. Also some preprocessing methods such as zero-loss alignment and visualization tools are available.
See the example notebooks on how to use the different functionalities of pyEELSMODEL.

Model based EELS quantification
+++++++++++++++++++++++++++++++
The main focus of pyEELSMODEL is on the quantification of core-loss via a model-based
approach. The first step is to chose a proper physical model which represents the experimental data.
In general, a model consists out of a background, cross sections and the low loss.
The optimal parameters can be found via a least squares or maximimum likelyhood minimization scheme

Philosophy
++++++++++
The model based approach tries to setup a correct/physical model for the experimental data.
The model contains the parameters of interest. For instance, the elemental
abundance can be retrieved from the core-loss cross sections.

pyEELSMODEL can be thought of as a toolbox to develop novel workflows to optimize
EELS processing in a transparent way. Moreover, due to the versatility of the methods,
the novel workflow can be compared to other ones in an easy way.

The workflows defined make it easier for the novice users to start quantifying their own
data.

pyEELSMODEL is a package which is written mainly in python language to make it
more readable and understandable this compromises in terms of speed but improves
understanding of how the data was treated (algorithms are no black box).

This package uses a minimal amount of dependencies (matplotlib, numpy, h5py and sci-kit)
which should make the package more stable over time without having conflicts when trying
to install other packages.

Structure of pyEELSMODEL
++++++++++++++++++++++++
The package of pyEELSMODEL contains a back-bone structure where spectrums, components
and fitters are defined. These classes are then used to perform operations on
the experimental data, the operations are defined as new classes.

Classes help understanding the procedure performed since inbetween results
can be saved as attributes. Also workflows with many input can be easier
modified as with function were many argumetns need to be passed with the
function.

Audience
++++++++
The pyEELSMODEL package is aimed at two types of users.

1. The electron microscopist who wants to analyze their EELS data. They can
use the workflows which should work in many cases.

2. The algorithm developers can use this package as reference to compare their
novel methods to existing algorithms.


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip <api.rst>`
