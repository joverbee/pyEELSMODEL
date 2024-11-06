.. pyEELSMODEL documentation master file, created by
   sphinx-quickstart on Mon Feb 26 11:38:01 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyEELSMODEL's documentation!
=======================================

.. toctree::
   intro
   installing
   Tutorial.ipynb
   CorelossExample.ipynb
   pyEELSMODEL
   acknowledgements
   :maxdepth: 1
   :caption: Contents:


Constained linear fitting via quadratic programming
===================================================

The new version includes constrained linear fitting which is used for both the linear
background model and fine structure in the model-based approach. More in depth information
be found in `Van den Broek et al. <https://doi.org/10.1016/j.ultramic.2023.113830>`_ and 
`Jannis et al. <https://arxiv.org/abs/2408.11870>`_. Following notebooks showcase how to 
use these novel methods.
1. InequalityBackground.ipynb: Shows how to use the constrained background model
2. ConstrainedFineStructure.ipynb: Shows how to use the constrained fine structure and 
its adavantage compared to unconstraiend
3. TbScO3Example.ipynb: Applies the novel methodologies to an experimental dataset. 