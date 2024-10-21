# aircraft_mapper

This repository is an initial one for code used to map aircraft data onto model (e.g. Community Atmosphere Model) grids.

"Flight-through-model" codes are frequently employed to map, or interpolate, model output onto an aircraft flight path in
order to to directly compare models and data. However, this exercise is really an extreme oversampling of coarse information
(e.g. $\Delta x = 100$ km climate model output, $\Delta x = 1$ km mesoscale model output) to compare with, relatively, very 
high resolution aircraft observations (e.g. $\approx 25000,\,250$ observations per grid climate, mesoscale model grid cell).
This small packages goes the opposite direction: it maps aircraft data onto model grids, grid cell by grid cell. This way,
statistics of observations can also be computed and directly compared with model parameters that model or parameterize
sub-grid-scale phenomena.

The input for this tool should be a complete description of a model grid and a flight data file. The model grid description
must include all information required to define
the full 3-D volume of every grid cell. While dynamical cores (e.g. Finite Volume, Spectral Element) that work with 
complex, unstructured grids (e.g. cubed sphere) with non-square grid cells (e.g. MPAS's Veronoi mesh) are becoming common,
often these products are conservativel regridded onto more user-friendly structured grids. To start, this tool will support
regular latitude, longitude grids, where only grid centers are needed to fully define the grid cell volumes. However,
algorithms exist to do this mapping onto grid cells of any shape, as long as those volumes are convex 
(e.g. [here](https://www.doc.ic.ac.uk/~dfg/graphics/graphics2008/GraphicsLecture04.pdf)).
