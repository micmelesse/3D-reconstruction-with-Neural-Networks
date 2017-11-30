ShapeNetCore data release v1 (July 2015)

These files contain ShapeNetCore: a densely annotated subset of ShapeNet released to the research community.  Each zip file is named by the synset noun offset in WordNet (version 3.0) as an eight-digit zero padded string.  For example, bench is contained within 02828884.zip since the WordNet synset offset for bench is 02828884 (you can browse WordNet at http://wordnetweb.princeton.edu/perl/webwn3.0).  The corresponding ImageNet synsets can be accessed at http://www.image-net.org/synset?wnid=n<synsetId> where <synsetId> is replaced by the synset offset (note that ImageNet includes an 'n' prefix for noun synsets).

Within each synset zip file there is a set of OBJ files of all the 3D models annotated under that synset.  Each model is under a directory named after its source id, which is the id of the original model on the online repository from which it was collected.  Within the model directory, you will find OBJ, MTL and texture image files.  You can view OBJ mesh files with software such as Assimp, the Open Asset Import Library (http://assimp.sourceforge.net/).

In addition to the model data, you will find <synsetId>.csv (comma-separated value format) files that contain metadata associated with the models in each synset.  The columns of these CSV files and their interpretation are as follows:

fullId : the unique id of the model
wnsynset : comma-separated list of WordNet synset offsets to which the model belongs
wnlemmas : comma-separated list of names (lemmas) of the WordNet synsets to which the model belongs
up : normalized vector in original model space coordinates indicating semantic "upright" orientation of model
front : normalized vector in original model space coordinates indicating semantic "front" orientation of model
name : name of the model as indicated on original model repository (uncurated)
tags : tags assigned to the model on original model repository (uncurated)

NOTE: The OBJ files have been pre-aligned so that the up direction is the +Y axis, and the front is the +X axis.  In addition each model is normalized to fit within a unit cube centered at the origin. The X-Y plane is the bilateral symmetry plane for most categories.

In addition to the existing CSV files, you can re-download the current metadata from the ShapeNet server using the get-metadata.sh shell script (requires bash and wget).

The taxonomy.json file contains a simple JSON format representation of the ShapeNetCore synset taxonomy indicating for each synset the synset offset (synsetId), the synset lemma (name), an array of the children synsets ids (children), and the total number of model instances (numInstances).  This JSON file is obtained from the more comprehensive ShapeNet taxonomy JSON at https://www.shapenet.org/resources/data/shapenetcore.taxonomy.json by filtering with the command:

jq "[.[] | recurse (.children[]?) | {synsetId: .metadata.name, name: .metadata.label, children: [.children[]?.metadata.name], numInstances: .metadata.numInstances }]" shapenetcore.taxonomy.json

(requires the JSON filter library jq which can be obtained from http://stedolan.github.io/jq/ , a linux x64 binary is included in this package)

If you use ShapeNet data you agree to abide by the ShapeNet terms of use (see terms-of-use.txt). You are only allowed to redistribute the data to your research associates and colleagues provided that they first agree to be bound by these terms and conditions.

For more information, please contact us at shapenet-webmaster@lists.stanford.edu

Last updated: 2017-08-28

CHANGELOG:

- 2017-08-28 Fix taxonomy.json URL

v1 (July 2015)
- Fixed missing MTL files for car synset (thanks to Alexey Dosovitskiy)
- Fixed geometry issues in many OBJ model files due to incomplete or incorrect triangulation from COLLADA files
- Add numInstances field to taxonomy.json for indicating number of models with metadata (corresponds to total number of rows in each synset's metadata .csv file)
- Removed popularity, nvertices and nfaces columns from metadata since they currently refer to stale or missing values

v0 (May 2015)
- Initial release
