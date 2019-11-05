## `templates`
In this directory, we store templates of experiment descriptions and processing
scripts such that updating them for each experiment is simple.

## Image analysis

For the image analysis pipeline we have 3 scripts and a standardized `README`
file that goes along all of the image analysis directories in
`src/image_analysis/`. Some of the scripts vary for experiments due to
differences in how the data was saved that day, meaning individual files or
stacks for all channels.

- `image_README.md` : standardized file to input information about the
  experiment.
- `image_metadata.py` : script that extracts the metadata from the directory
  name such that the user doesn't have to input anything into the other
  scripts.
- `image_processing.py` : script that process the raw `tif` files from each
  experiment. Segments the cells using the **mCherry** channel, and extracts
  the single-cell intensities.
- `image_analysis.py` : script that takes the output from the
  `image_processing.py` pipeline, applies filters on the area and eccentricity
  of cells and outputs a filtered data frame along with some diagnostic plots.