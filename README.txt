ShapeNetSem data release June 2015 (v0)

These files contain ShapeNetSem: a subset of ShapeNet richly annotated with physical attributes, which we release for the benefit of the research community.

There are several pieces of model data that are available:
- models-OBJ.zip : OBJ format 3D mesh files (with accompanying MTL material definition files)
- models-textures.zip : texture files used by above 3D mesh representations
- models-COLLADA.zip : COLLADA (DAE) format 3D mesh files
- models-binvox.zip : Binary voxelizations of model surfaces in binvox format
- models-binvox-solid.zip : Filled-in binary voxelizations of models in binvox format
- models-screenshots.zip : Pre-rendered screenshots of each model from 6 canonical orientations (front, back, left, right, bottom, top), and another 6 "turn table" positions around the model

In addition to the model data, you will find a metadata.csv (comma-separated value format) file that contains the metadata associated with each model.  The columns of this file and their interpretation are as follows:

fullId : the unique id of the model
category : manually annotated categories to which this model belongs
wnsynset : comma-separated list of WordNet synset offsets to which the model belongs
wnlemmas : comma-separated list of names (lemmas) of the WordNet synsets to which the model belongs
up : normalized vector in original model space coordinates indicating semantic "upright" orientation of model
front : normalized vector in original model space coordinates indicating semantic "front" orientation of model
unit : scale unit converting model virtual units to meters
aligned.dims : aligned dimensions of model after rescaling to meters and upright-front realignment (X-right, Y-back, Z-up)
isContainerLike : whether this model is container-like (i.e., is internally empty)
surfaceVolume : total volume of surface voxelization of mesh (m^3) used for container-like objects
solidVolume : total volume of solid (filled-in) voxelization of mesh (m^3) used for non container-like objects
supportSurfaceArea : surface area of support surface (usually bottom) (m^2)
weight : estimated weight of object in Kg computed from material priors and appropriate volume
staticFrictionForce : static friction force required to push object computed from supportSurfaceArea and coefficient of static friction using material priors
name : name of the model as indicated on original model repository (uncurated)
tags : tags assigned to the model on original model repository (uncurated)

In addition to the existing CSV file, you can re-download the current metadata from the ShapeNet server through the following URL:
https://www.shapenet.org/solr/models3d/select?q=isAligned%3Atrue+AND+source%3Awss+AND+category%3A*&rows=100000&fl=fullId%2Ccategory%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cunit%2Caligned.dims%2CisContainerLike%2CsurfaceVolume%2CsolidVolume%2CsupportSurfaceArea%2Cweight%2CstaticFrictionForce%2Cname%2Ctags&wt=csv&indent=true

Please note that the above link restricts the retrieved metadata to only models that have manually verified alignments and categorizations (currently about half of the total). If you would like the full list of all models, use the following link: 
https://www.shapenet.org/solr/models3d/select?q=source%3Awss&rows=100000&fl=fullId%2Ccategory%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cunit%2Caligned.dims%2CisContainerLike%2CsurfaceVolume%2CsolidVolume%2CsupportSurfaceArea%2Cweight%2CstaticFrictionForce%2Cname%2Ctags&wt=csv&indent=true

There are several other secondary metadata files:
- categories.synset.csv : maps manual category labels to WordNet synsets and glosses
- materials.csv : set of per-category material priors extracted from OpenSurfaces [Bell et al. 2014] dataset
- densities.csv : material densities (g / cm3) and static coefficients of friction
- taxonomy.txt : defines the taxonomy of our manual categories. Lines start with parent category followed by child categories, all separated by tabs. Note comment lines starting with '#'

To get the above metadata files, just replace the last part of the URL to this README.txt file with the filename of the metadata file.

If you use ShapeNet data you agree to abide by the ShapeNet terms of use (see terms-of-use.txt). You are only allowed to redistribute the data to your research associates and colleagues provided that they first agree to be bound by these terms and conditions.

Please cite the main ShapeNet technical report (information to be provided in the near future) and the "Semantically-enriched 3D Models for Common-sense Knowledge" workshop paper.

For more information, please contact us at shapenet-webmaster@lists.stanford.edu and indicate ShapeNetSem in the title of your email.

Last updated: 2017-01-24

Change log:
- 2017-01-24 : Fix metadata links and clarify locaiton of metadata files

