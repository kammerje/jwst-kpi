# Kernel Phase Pipeline for JWST

Pipeline step and example scripts to process kernel phase data with JWST.

This is a rewrite of the kernel phase pipeline from @kammerje
(https://github.com/kammerje/xara/tree/develop), which was implemented as a xara
fork with modifications for the JWST pipeline. The main goal is to have a
separate repository from xara, but that uses xara as a dependency. Another goal
is to reimplement the same functionality but using the standard JWST pipeline
structure following the [`stpipe`
documentation](https://jwst-pipeline.readthedocs.io/en/latest/jwst/stpipe/devel_step.html).
