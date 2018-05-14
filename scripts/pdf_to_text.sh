convert -density 300 $1.pdf -depth 8 $1.tiff
tesseract $1.tiff $1