# Documentation Acess
To access the documentation, simply [click here](https://github.com/brunofaria27/image-processing/blob/main/documentation/Documentation%20-%20Portuguese.pdf). However, at the moment we only have the documentation in Portuguese in the form of an article. Nothing to add to the README.


## CRIC Cervix Cell Classification - [CSV Description](https://github.com/brunofaria27/image-processing/blob/main/src/data/classifications.csv)

400 images from microscope slides of the uterine cervix using the conventional smear (Pap smear) and the epithelial cell abnormalities classified according to Bethesda system.

### Data Fields
- `image_id`
  This is the integer that identifies the image at http://database.cric.com.br/.
- `image_filename`
  This is the name that identifies the image in the ZIP file that you have.
- `image_doi`
  This is the DOI that identifies the image.
- `cell_id`
  This is the integer that identifies the cell at http://database.cric.com.br/.
- `bethesda_system`
  Classification of the cell
  using the Bethesda system.
  It is on of the following:
  - Negative for intraepithelial lesion
  - ASC-US
    Atypical squamous cells of undetermined significance
  - ASC-H
    Atypical squamous cells cannot exclude HSIL
  - LSIL
    Low grade squamous intraepithelial lesion
  - HSIL
    High grade squamous intraepithelial lesion
  - SCC
    Squamous cell carcinoma
- `nucleus_x`
  Integer between 1 and 1384 equal to coordinate x of the pixel that represent the cell.
- `nucleus_y`
  Integer between 1 and 1384 equal to coordinate y of the pixel that represent the cell.
