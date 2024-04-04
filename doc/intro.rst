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
After the model is created, the optimal parameters can be found via a least squares
or maximimum likelihood minimization scheme.
Via the Cramer-Rao bound, a lower bound on the error can be provided which can be
used as estimate for the error on the different parameters. The Coreloss Example
notebook shows how the model-based approach is implemented in pyEELSMODEL.
See `Verbeeck et al. <https://github.com/joverbee/eelsmodel>`_ for more information on
the methodology.


Philosophy
++++++++++
pyEELSMODEL can be thought of as a toolbox to develop novel workflows to optimize
EELS processing in a transparent way. Moreover, due to the versatility of the methods,
the novel workflow can be compared to other ones in an easy way.
For the more novice users, pyEELSMODEL provide workflows to perform reliable and
reproducable EELS quantification. Moreover, the structure of the code is mainly
written in python to make the algorithms more understandable.
This compromises in terms of speed but improves understanding of how the
data was treated (algorithms are no black box).

This package uses a minimal amount of dependencies (matplotlib, numpy, h5py and sci-kit)
which should make the package more stable over time without having conflicts when trying
to install other packages.

Structure of pyEELSMODEL
++++++++++++++++++++++++
The package of pyEELSMODEL contains a back-bone structure where spectrums, components
and fitters are defined. These classes are re-used to perform operations on
the experimental data, these operations are defined as new classes. For instance,
the BackgroundRemoval class performs the workflow of background subtraction which
first finds the initial parameters and next uses these to perform the non-linear
optimization.

Classes facilitate comprehension of the procedures undertaken by allowing intermediate
results to be stored as attributes. Additionally, workflows involving numerous inputs
can be more readily modified compared to functions where many arguments must be passed explicitly.

Audience
++++++++
The pyEELSMODEL package is aimed at two types of users.

1. The electron microscopist who wants to analyze their EELS data. They can
use the workflows which should work in many cases. If the standard settings fail,
it is relatively easy to modify them and try other input parameters.

2. The algorithm developers can use this package as reference to compare their
novel methods to existing algorithms and even add novel algorithms to this
package.


